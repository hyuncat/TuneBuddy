import qdarktheme

class AppConfig:
    """
    Configuration File
    """
    APP_NAME: str = "virtuOS"
    SAMPLE_RATE: int = 44100
    FRAME_SIZE: int = 2048
    HOP_SIZE: int = 128
    
    # Instrument settings: Violin
    MIN_VIOLIN_FREQ = 196
    TUNING = 0.0

    # Number of audio channels to record/playback from
    CHANNELS: int = 1 # Mono, for simplicity (also, idrk)

    # Default to False, will be updated in initialize() based on the environment
    RUNNING_IN_JUPYTER: bool = False

    DARK_THEME: bool = True

    @classmethod
    def initialize(cls, app) -> None:
        """
        Perform any necessary initializations here, e.g.:
        - Loading settings from a file
        - Determining the runtime environment
        """
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                cls.RUNNING_IN_JUPYTER = True
        except ImportError:
            # IPython is not installed, so we're not running in a Jupyter environment
            cls.RUNNING_IN_JUPYTER = False

        if cls.DARK_THEME:
            app.setStyleSheet(qdarktheme.load_stylesheet("dark"))