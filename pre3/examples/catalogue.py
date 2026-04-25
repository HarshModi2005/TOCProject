"""
Catalogue of canonical examples.  Re-exports each example module.

Use as a registry for tests and demos:

    from pre3.examples.catalogue import REGISTRY
    for entry in REGISTRY:
        print(entry["name"], "→", entry["class"])
"""

from . import anbn, anbncn, dyck2, wcwR, wwR

REGISTRY = [
    {
        "name": "aⁿbⁿ",
        "class": "DCFL",
        "module": anbn,
        "is_cfl": True,
        "is_dcfl": True,
        "lrk_buildable": True,
    },
    {
        "name": "wcwᴿ",
        "class": "DCFL",
        "module": wcwR,
        "is_cfl": True,
        "is_dcfl": True,
        "lrk_buildable": True,
    },
    {
        "name": "wwᴿ",
        "class": "CFL ∖ DCFL",
        "module": wwR,
        "is_cfl": True,
        "is_dcfl": False,
        "lrk_buildable": False,   # MUST fail with LR conflicts
    },
    {
        "name": "Dyck-2",
        "class": "DCFL",
        "module": dyck2,
        "is_cfl": True,
        "is_dcfl": True,
        "lrk_buildable": True,
    },
    {
        "name": "aⁿbⁿcⁿ",
        "class": "non-CFL",
        "module": anbncn,
        "is_cfl": False,
        "is_dcfl": False,
        "lrk_buildable": False,    # has no grammar at all
    },
]
