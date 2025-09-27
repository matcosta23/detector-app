from typing import List, Dict, Optional

def apply_class_map(dets: List[dict], classes_map: Optional[Dict[int, str]]):
    if not classes_map:
        return dets
    for d in dets:
        if d.get("cls") in classes_map:
            d["label"] = classes_map[d["cls"]]
    return dets