class PlotInternals:
    """
    Collection of matplotlib objects needed to allow refresh of a plot with
    new data.
    """

    def __init__(self):
        self.img = None
        self.cbar = None
        self.mappable = None
