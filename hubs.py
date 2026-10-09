"""
Grouping of the 90 AAL regions into hubs (larger anatomical units) for the hub
analysis in s08_hubs.py. Each hub is defined once here and made separately for
the left and right hemisphere (e.g. "Mesial temporal L", "Mesial temporal R").

Edit HUB_REGIONS to change the grouping. Every AAL region must belong to exactly
one hub; s08 stops with a list of any region that is missing or listed twice.
Names are the AAL (SPM12) base names without the _L / _R suffix.
"""

# Hub name -> AAL base names. Order here is the order used in tables and figures.
HUB_REGIONS = {
    # Temporal lobe
    "Mesial temporal":      ["Hippocampus", "ParaHippocampal", "Amygdala"],
    "Temporal pole":        ["Temporal_Pole_Sup", "Temporal_Pole_Mid"],
    "Lateral temporal":     ["Temporal_Sup", "Temporal_Mid", "Heschl"],
    "Basal temporal":       ["Temporal_Inf", "Fusiform"],
    # Frontal lobe
    "Orbitofrontal":        ["Frontal_Sup_Orb", "Frontal_Mid_Orb", "Frontal_Inf_Orb",
                             "Frontal_Med_Orb", "Rectus", "Olfactory"],
    "Dorsolateral frontal": ["Frontal_Sup", "Frontal_Mid"],
    "Inferior frontal":     ["Frontal_Inf_Oper", "Frontal_Inf_Tri"],
    "Medial frontal":       ["Frontal_Sup_Medial", "Supp_Motor_Area", "Cingulum_Ant"],
    # Pericentral and insula
    "Pericentral":          ["Precentral", "Postcentral", "Paracentral_Lobule",
                             "Rolandic_Oper"],
    "Insula":               ["Insula"],
    # Parietal lobe
    "Superior parietal":    ["Parietal_Sup"],
    "Inferior parietal":    ["Parietal_Inf", "SupraMarginal", "Angular"],
    "Medial parietal":      ["Precuneus", "Cingulum_Post", "Cingulum_Mid"],
    # Remaining regions, kept so the network stays complete
    "Occipital":            ["Calcarine", "Cuneus", "Lingual", "Occipital_Sup",
                             "Occipital_Mid", "Occipital_Inf"],
    "Thalamus":             ["Thalamus"],
    "Basal ganglia":        ["Caudate", "Putamen", "Pallidum"],
}


def split_hemisphere(roi_name):
    """'Hippocampus_L' -> ('Hippocampus', 'L'). Raises ValueError if no _L/_R suffix."""
    for side in ("L", "R"):
        if roi_name.endswith("_" + side):
            return roi_name[:-2], side
    raise ValueError(f"AAL region name without _L/_R suffix: {roi_name}")


def assign_hubs(roi_names):
    """
    Map each ROI (in the given order) to a hemisphere-specific hub.
    Returns (hub_names, members): hub_names is the ordered list of hubs actually
    present, and members[hub] the list of ROI indices in it.
    """
    lookup = {}
    for hub, bases in HUB_REGIONS.items():
        for base in bases:
            if base in lookup:
                raise ValueError(f"{base} is listed in both {lookup[base]} and {hub}")
            lookup[base] = hub

    members, unassigned = {}, []
    for i, roi in enumerate(roi_names):
        base, side = split_hemisphere(str(roi))
        if base not in lookup:
            unassigned.append(str(roi))
            continue
        members.setdefault(f"{lookup[base]} {side}", []).append(i)
    if unassigned:
        raise ValueError(f"AAL regions not assigned to any hub in hubs.py: {unassigned}")

    # Order: hubs in HUB_REGIONS order, left hemisphere before right.
    hub_names = [f"{h} {s}" for h in HUB_REGIONS for s in ("L", "R") if f"{h} {s}" in members]
    return hub_names, members
