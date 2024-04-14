import json
import os
import pathlib
from functools import partial
import warnings
import traceback


import pandas as pd
import torch.multiprocessing as mp
from joblib import Memory
from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config
from utils import format_dict, seed_everything
import datasets

# See https://github.com/pytorch/pytorch/issues/11201, https://github.com/pytorch/pytorch/issues/973
# Not for dataloader, but for multiprocessing batches
mp.set_sharing_strategy('file_system')
queue_results = None

cache = Memory('cache/' if config.use_cache else None, verbose=0)
runs_dict = {}
seed_everything()
console = Console(highlight=False)


def my_collate(batch):
    # Avoid stacking images (different size). Return everything as a list
    to_return = {k: [d[k] for d in batch] for k in batch[0].keys()}
    return to_return


def run_program(parameters, queues_in_, input_type_, retrying=True):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno
    from video_segment import VideoSegment

    global queue_results

    code, sample_id, image, possible_answers, query = parameters

    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query, ' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match, code):\n' \
                  f'    # Answer is:'

    code = code.replace('```', '').replace('python', '')
    code = code_header + code.strip()

    print(code)

    answer = None
    reason = None
    info = None
    compilation_error = ''
    runtime_error = ''

    try:
        exec(compile(code, 'Codex', 'exec'), globals())
    except Exception as e:
        compilation_error = str(e)
        print(f'Sample {sample_id} failed at compilation time with error: {e}')
        # try:
        #     with open(config.fixed_code_file, 'r') as f:
        #         fixed_code = f.read()
        #     code = code_header + fixed_code
        #     exec(compile(code, 'Codex', 'exec'), globals())
        # except Exception as e2:
        #     print(f'Not even the fixed code worked. Sample {sample_id} failed at compilation time with error: {e2}')
        #     return None, code

    queues = [queues_in_, queue_results]

    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)

    try:
        answer, reason, info = globals()[f'execute_command_{sample_id}'](
            # Inputs to the function
            image, possible_answers, query,
            # Classes to be used
            image_patch_partial, video_segment_partial,
            # Functions to be used
            llm_query_partial, bool_to_yesno, distance, best_image_match, code)
    except Exception as e:
        # print full traceback
        traceback.print_exc()
        # print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        runtime_error = str(e)
        # if retrying:
        #     return None, code
        # Retry again with fixed code
        # new_code = "["  # This code will break upon execution, and it will be caught by the except clause
        # result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_,
        #                      retrying=True)[0]

    # The function run_{sample_id} is defined globally (exec doesn't work locally). A cleaner alternative would be to
    # save it in a global dict (replace globals() for dict_name in exec), but then it doesn't detect the imported
    # libraries for some reason. Because defining it globally is not ideal, we just delete it after running it.
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']  # If it failed to compile the code, it won't be defined
    return {
        'answer': answer,
        'compilation_error': compilation_error,
        'runtime_error': runtime_error,
        'info': info,
        'code': code,
        'reason': reason,
    }


def worker_init(queue_results_):
    global queue_results
    index_queue = mp.current_process()._identity[0] % len(queue_results_)
    queue_results = queue_results_[index_queue]


def main():
    mp.set_start_method('spawn')

    from vision_processes import queues_in, finish_all_consumers, forward, manager
    from datasets import get_dataset

    batch_size = config.dataset.batch_size
    num_processes = min(batch_size, 50)

    if config.multiprocessing:
        queue_results_main = manager.Queue()
        queues_results = [manager.Queue() for _ in range(batch_size)]
    else:
        queue_results_main = None
        queues_results = [None for _ in range(batch_size)]

    model_name_codex = 'codellama' if config.codex.model == 'codellama' else 'codex'
    codex = partial(forward, model_name=model_name_codex, queues=[queues_in, queue_results_main])

    if config.clear_cache:
        cache.clear()

    if config.wandb:
        import wandb
        wandb.init(project="viper", config=OmegaConf.to_container(config))
        # log the prompt file
        wandb.save(config.codex.prompt)

    dataset = get_dataset(config.dataset)

    codes_all = None
    if config.use_cached_codex:
        results = pd.read_csv(config.cached_codex_path)
        codes_all = [r.split('# Answer is:')[1] for r in results['code']]
    # python -c "from joblib import Memory; cache = Memory('cache/', verbose=0); cache.clear()"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=my_collate)
    input_type = dataset.input_type

    all_answers = []
    all_infos = []
    all_compilation_errors = []
    all_runtime_errors = []
    all_codes = []
    all_reasons = []
    # all_reflection = []
    all_potential_issues = []
    all_revised_codes = []
    # anytool result
    all_tool_codes = []

    all_groundtruths = []
    all_ids = []
    all_queries = []
    all_img_paths = []
    all_possible_answers = []
    all_query_types = []

    with mp.Pool(processes=num_processes, initializer=worker_init, initargs=(queues_results,)) \
            if config.multiprocessing else open(os.devnull, "w") as pool:
        try:
            n_batches = len(dataloader)

            for i, batch in tqdm(enumerate(dataloader), total=n_batches):

                # Combine all queries and get Codex predictions for them
                # TODO compute Codex for next batch as current batch is being processed

                for sample_id, img, possible_answers, query, gt, extra_context, code, issue, tool_spec, dfs_filename, revised_code \
                        in zip(batch['sample_id'], batch['image'], batch['possible_answers'], batch['query'], batch['answer'], batch['extra_context'], batch['code'], batch['issue'], batch['tool_spec'], batch['dfs_filename'], batch['revised_code']):
                    # code -> str, message -> list[dict[str, str]]
                    _, message = codex(prompt=query, input_type=input_type, extra_context=extra_context)
                    execution_result = run_program([code, sample_id, img, possible_answers, query], queues_in, input_type)
                    execution_result['groundtruth'] = gt
                    # potential_issues, revised_code, message = forward('reflection', code, message, execution_result)
                    _, _, message = forward('reflection', code, message, execution_result, issue)
                    execution_result['potential_issues'] = issue
                    execution_result['revised_code'] = revised_code

                    tool_code = forward('anytool', message, revised_code, tool_spec, dfs_filename)
                    execution_result['tool_code'] = tool_code
                    all_tool_codes.append(execution_result['tool_code'])

                    all_answers.append(execution_result['answer'])
                    all_infos.append(json.dumps(execution_result['info'], indent=2))
                    all_codes.append(execution_result['code'])
                    all_compilation_errors.append(execution_result['compilation_error'])
                    all_runtime_errors.append(execution_result['runtime_error'])
                    all_reasons.append(execution_result['reason'])
                    all_potential_issues.append(execution_result['potential_issues'])
                    all_revised_codes.append(execution_result['revised_code'])
                    all_ids.append(sample_id)
                    all_groundtruths.append(gt)
                    all_possible_answers.append(possible_answers)
                    all_queries.append(query)

                    # for r in tqdm(range(1, config.num_reflections), desc='Reflections'):
                    #     code = revised_code
                    #     execution_result = run_program([code, sample_id, img, possible_answers, query], queues_in, input_type)
                    #     execution_result['groundtruth'] = gt
                    #     potential_issues, revised_code = forward('reflection', code, message, execution_result)
                    #     execution_result['potential_issues'] = potential_issues
                    #     execution_result['revised_code'] = revised_code

                    #     all_answers.append(execution_result['answer'])
                    #     all_infos.append(json.dumps(execution_result['info'], indent=2))
                    #     all_codes.append(execution_result['code'])
                    #     all_compilation_errors.append(execution_result['compilation_error'])
                    #     all_runtime_errors.append(execution_result['runtime_error'])
                    #     all_reasons.append(execution_result['reason'])
                    #     all_potential_issues.append(execution_result['potential_issues'])
                    #     all_revised_codes.append(execution_result['revised_code'])
                    #     all_ids.append(sample_id + f'_{r}')
                    #     all_groundtruths.append(gt)
                    #     all_possible_answers.append(possible_answers)
                    #     all_queries.append(query)
        except Exception as e:
            # print full stack trace
            traceback.print_exc()
            console.print(f'Exception: {e}')
            console.print("Completing logging and exiting...")

    try:
        accuracy = dataset.accuracy(all_answers, all_groundtruths, all_possible_answers, all_query_types)
        console.print(f'Final accuracy: {accuracy}')
    except Exception as e:
        print(f'Error computing accuracy: {e}')

    if config.save:
        results_dir = pathlib.Path(config['results_dir'])
        results_dir = results_dir / config.dataset.split
        results_dir.mkdir(parents=True, exist_ok=True)
        if not config.save_new_results:
            filename = 'results.csv'
        else:
            existing_files = list(results_dir.glob('results_*.csv'))
            if len(existing_files) == 0:
                filename = 'results_0.csv'
            else:
                filename = 'results_' + str(max([int(ef.stem.split('_')[-1]) for ef in existing_files if
                                                 str.isnumeric(ef.stem.split('_')[-1])]) + 1) + '.csv'
        print('Saving results to', filename)
        df = pd.DataFrame([all_answers, 
                           all_groundtruths, 
                           all_ids, 
                           all_queries, 
                        #    all_img_paths,
                           all_possible_answers,
                           all_codes,
                           all_revised_codes,
                           all_tool_codes,
                           all_infos,
                           all_reasons,
                        #    all_reflection,
                           all_potential_issues,
                           all_compilation_errors,
                           all_runtime_errors]).T
        df.columns = [
            'answer', 
            'groundtruth', 
            'id', 
            'query', 
            # 'img_path', 
            'possible_answers', 
            'code', 
            'revised_code', 
            'tool_code',
            'info', 
            'reason', 
            # 'reflection',
            'potential_issues',
            'compilation_error', 
            'runtime_error'
            ]
        # make the result column a string
        df.to_csv(results_dir / filename, header=True, index=False, encoding='utf-8', sep='|')
        # torch.save([all_results, all_answers, all_codes, all_ids, all_queries, all_img_paths], results_dir/filename)

        if config.wandb:
            wandb.log({'accuracy': accuracy})
            wandb.log({'results': wandb.Table(dataframe=df, allow_mixed_types=True)})

    finish_all_consumers()


if __name__ == '__main__':
    main()
