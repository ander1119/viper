import math

class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left, lower, right, upper : int
        An int describing the position of the (left/lower/right/upper) border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None, to_yesno: bool=False)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer to "What is this?".
        If to_yesno is set to True, the answer must contain 'yes' or 'no'
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    llm_query(question: str, to_yesno: bool=False)->str
        Returns the answer to a basic question which is unrelevant to the image. 
        If to_yesno is set to True, the answer must contain 'yes' or 'no'
    """

    def __init__(self, image, left: int = None, lower: int = None, right: int = None, upper: int = None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.
        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left, lower, right, upper : int
            An int describing the position of the (left/lower/right/upper) border of the crop's bounding box in the original image.
        """
        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, lower:upper, left:right]
            self.left = left
            self.upper = upper
            self.right = right
            self.lower = lower

        self.width = self.cropped_image.shape[2]
        self.height = self.cropped_image.shape[1]

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

    def get_subtitles(self) -> List[str]:
        """Return a list of str, which is subtitles that present in the current frame 
        Examples
        --------
        >>> subtiltles = '\n'.join(image_patch.get_subtitles())
        >>> caption = image_patch.simple_query(f"What's happening in the scene which has subtitles {subtiltles}?")
        """

    def find(self, object_name: str) -> List[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop

        Examples
        --------
        >>> # return the foo
        >>> def execute_command(image) -> List[ImagePatch]:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patches = image_patch.find("foo")
        >>>     return foo_patches
        """
        return find_in_image(self.cropped_image, object_name)

    def exists(self, object_name: str) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.

        Examples
        -------
        >>> # Are there both foos and garply bars in the photo?
        >>> def execute_command(image)->str:
        >>>     image_patch = ImagePatch(image)
        >>>     is_foo = image_patch.exists("foo")
        >>>     is_garply_bar = image_patch.exists("garply bar")
        >>>     return bool_to_yesno(is_foo and is_garply_bar)
        """
        return len(self.find(object_name)) > 0

    def best_text_match(self, option_list: List[str], prefix: str=None) -> str:
        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options

        Examples
        -------
        >>> # Is the foo gold or white?
        >>> def execute_command(image)->str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patches = image_patch.find("foo")
        >>>     # Question assumes one foo patch
        >>>     return foo_patches[0].best_text_match(["gold", "white"])
        """
        return best_text_match(self.cropped_image, option_list, prefix)

    def simple_query(self, question: str = None, to_yesno=False) -> str:
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        to_yesno : bool
            A boolean indicate whether answer must contain yes/no
        Examples
        -------

        >>> # What color is the foo?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patches = image_patch.find("foo")
        >>>     foo_patch = foo_patches[0]
        >>>     return foo_patch.simple_query("What is the color?")

        >>> # Is the second bar from the left quuxy?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     bar_patches = image_patch.find("bar")
        >>>     bar_patches.sort(key=lambda x: x.horizontal_center)
        >>>     bar_patch = bar_patches[1]
        >>>     return bar_patch.simple_query("Is the bar quuxy?", to_yesno=True)
        """
        return simple_query(self.cropped_image, question)

    def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:
        """Returns a new ImagePatch cropped from the current ImagePatch.
        Parameters
        -------
        left, lower, right, upper : int
            The (left/lower/right/upper)most pixel of the cropped image.
        -------
        """
        return ImagePatch(self.cropped_image, left, lower, right, upper)

    def overlaps_with(self, left, lower, right, upper):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left, lower, right, upper : int
            the (left/lower/right/upper) border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False

        Examples
        --------
        >>> # black foo on top of the qux
        >>> def execute_command(image) -> ImagePatch:
        >>>     image_patch = ImagePatch(image)
        >>>     qux_patches = image_patch.find("qux")
        >>>     qux_patch = qux_patches[0]
        >>>     foo_patches = image_patch.find("black foo")
        >>>     for foo in foo_patches:
        >>>         if foo.vertical_center > qux_patch.vertical_center
        >>>             return foo
        """
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower

    def llm_query(self, question: str, to_yesno: bool=False)->str:
        '''Answers a text question using GPT-3. The input question is always a formatted string with a variable in it.
        
        Parameters
        ----------
        question: str
            the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
        to_yesno : bool
            A boolean indicate whether answer must contain yes/no
        '''


def best_image_match(list_patches: List[ImagePatch], content: List[str], return_index=False) -> Union[ImagePatch, int]:
    """Returns the patch most likely to contain the content.
    Parameters
    ----------
    list_patches : List[ImagePatch]
    content : List[str]
        the object of interest
    return_index : bool
        if True, returns the index of the patch most likely to contain the object

    Returns
    -------
    int
        Patch most likely to contain the object
    """
    return best_image_match(list_patches, content, return_index)


def distance(patch_a: ImagePatch, patch_b: ImagePatch) -> float:
    """
    Returns the distance between the edges of two ImagePatches. If the patches overlap, it returns a negative distance
    corresponding to the negative intersection over union.

    Parameters
    ----------
    patch_a : ImagePatch
    patch_b : ImagePatch

    Examples
    --------
    # Return the qux that is closest to the foo
    >>> def execute_command(image):
    >>>     image_patch = ImagePatch(image)
    >>>     qux_patches = image_patch.find('qux')
    >>>     foo_patches = image_patch.find('foo')
    >>>     foo_patch = foo_patches[0]
    >>>     qux_patches.sort(key=lambda x: distance(x, foo_patch))
    >>>     return qux_patches[0]
    """
    return distance(patch_a, patch_b)


def bool_to_yesno(bool_answer: bool) -> str:
    return "yes" if bool_answer else "no"


def coerce_to_numeric(string):
    """
    This function takes a string as input and returns a float after removing any non-numeric characters.
    If the input string contains a range (e.g. "10-15"), it returns the first value in the range.
    """
    return coerce_to_numeric(string)


class VideoSegment:
    """A Python class containing a set of frames represented as ImagePatch objects, as well as relevant information.
    Attributes
    ----------
    video : torch.Tensor
        A tensor of the original video.
    start : int
        An int describing the starting frame in this video segment with respect to the original video.
    end : int
        An int describing the ending frame in this video segment with respect to the original video.
    num_frames->int
        An int containing the number of frames in the video segment.

    Methods
    -------
    face_identify(image: ImagePatch) -> str
        Return an unique identifier according to person in image
    select_answer(self, info: dict, question: str, options: List[str]) -> (str, str):
        Return (answer, reason) for the question and options according to given information
    trim(start, end) -> VideoSegment
        Returns a new VideoSegment containing a trimmed version of the original video at the [start, end] segment.
    frame_iterator() -> Iterator[ImagePatch]
        Returns an iterator over the frames in the video segment.
    """

    def __init__(self, video: torch.Tensor, start: int = None, end: int = None, parent_start=0, queues=None):
        """Initializes a VideoSegment object by trimming the video at the given [start, end] times and stores the
        start and end times as attributes. If no times are provided, the video is left unmodified, and the times are
        set to the beginning and end of the video.

        Parameters
        -------
        video : torch.Tensor
            A tensor of the original video.
        start : int
            An int describing the starting frame in this video segment with respect to the original video.
        end : int
            An int describing the ending frame in this video segment with respect to the original video.
        """

        if start is None and end is None:
            self.trimmed_video = video
            self.start = 0
            self.end = video.shape[0]  # duration
        else:
            self.trimmed_video = video[start:end]
            if start is None:
                start = 0
            if end is None:
                end = video.shape[0]
            self.start = start + parent_start
            self.end = end + parent_start

        self.num_frames = self.trimmed_video.shape[0]

    def face_identify(self, image: ImagePatch) -> Union[str, None]:
        """Returns unique identifier as string for the person in the image, return None if there's no person or the person is unidentifiable

        Examples
        -------
        >>> def execute_command(video)->bool:
        >>>     role_info = {}
        >>>     video_segment = VideoSegment(video)
        >>>     for i, frame in enumerate(video_segment.frame_iterator()):
        >>>         for person in frame.find("person"):
        >>>             person_id = video_segment.face_identify(person)    
        >>>             if person_id is not None:
        >>>                 person_action = person.simple_query("What's he/she doing?")
        >>>                 if person_id in role_info:
        >>>                     role_info[person_id].append(person_action)
        >>>                 else:
        >>>                     role_info[person_id] = [person_action]
        """

    def frame_from_index(self, index) -> ImagePatch:
        """Returns the frame at position 'index', as an ImagePatch object.

        Examples
        -------
        >>> # Is there a foo in the frame bar appears?
        >>> def execute_command(video)->bool:
        >>>     video_segment = VideoSegment(video)
        >>>     for i, frame in enumerate(video_segment.frame_iterator()):
        >>>         if frame.exists("bar"):
        >>>             frame_after = video_segment.frame_from_index(i+1)
        >>>             return frame_after.exists("foo")
        """
        return ImagePatch(self.trimmed_video[index])

    def trim(self, start: Union[int, None] = None, end: Union[int, None] = None) -> VideoSegment:
        """Returns a new VideoSegment containing a trimmed version of the original video at the [start, end]
        segment.

        Parameters
        ----------
        start : Union[int, None]
            An int describing the starting frame in this video segment with respect to the original video.
        end : Union[int, None]
            An int describing the ending frame in this video segment with respect to the original video.

        Examples
        --------
        >>> # Return the second half of the video
        >>> def execute_command(video):
        >>>     video_segment = VideoSegment(video)
        >>>     video_second_half = video_segment.trim(video_segment.num_frames // 2, video_segment.num_frames)
        >>>     return video_second_half
        """
        if start is not None:
            start = max(start, 0)
        if end is not None:
            end = min(end, self.num_frames)

        return VideoSegment(self.trimmed_video, start, end, self.start)

    def select_answer(self, info: dict, question: str, options: List[str]) -> (str, str):
        """Return (answer, reason) for the question and options according to given information"""
        return select_answer(self.trimmed_video, info, question, options)

    def frame_iterator(self) -> Iterator[ImagePatch]:
        """Returns an iterator over the frames in the video segment.

        Examples
        -------
        >>> # Return the frame when the kid kisses the cat
        >>> def execute_command(video):
        >>>     video_segment = VideoSegment(video, annotation)
        >>>     for i, frame in enumerate(video_segment.frame_iterator()):
        >>>         subtiltles = frame.get_subtitles()
        >>>         if frame.exists("kid") and frame.exists("cat") and "yes" in frame.simple_query("Is the kid kissing the cat?", to_yesno=True):
        >>>             return frame
        """
        for i in range(self.num_frames):
            yield self.frame_from_index(i)