import torch
import _LAZY

class lazy_execute(object):      
    def __init__(self, enabled=True):          
        self._enabled = enabled

    def __enter__(self):                                                                                                 
        _LAZY._enable_lazy_mode()
        #print("Enable Lazy Mode!")

    def __exit__(self, type, value, traceback):         
        _LAZY._disable_lazy_mode()
        #print("Disable Lazy Mode!")

