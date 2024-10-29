import os.path as osp

class Config():
    
    def __init__(self):
        
        self.project_path = osp.dirname(osp.abspath(__file__))
        
        # Dataset
        self.dataset = 'refcoco'
        self.ref_base = 'PATH'
        self.ref_dir = osp.join(self.ref_base, self.dataset)
        self.coco_dir = 'PATH'
        self.panoptic_anns = osp.join(self.coco_dir, 'annotations')