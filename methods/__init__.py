from .avg import merge_avg
from .dare import merge_dare_linear, merge_dare_ties
from .adamerging import merge_adamerging
from .breadcrumbs import merge_breadcrumbs
from .fisher import merge_fisher
from .from_merge import merge_from
from .free_merge import merge_free
from .iso import merge_iso_c, merge_iso_cts
from .model_stock import merge_model_stock
from .lamp_merge import merge_lamp_merge
from .lamp_merge_analysis import merge_lamp_merge as merge_lamp_merge_analysis
from .regmean import merge_regmean
from .robustmerge import merge_robustmerge
from .ties import merge_ties

METHOD_ALIASES = {
    'avg': 'avg',
    'ties': 'ties',
    'dare_linear': 'dare_linear',
    'dare-liner': 'dare_linear',
    'dare_liner': 'dare_linear',
    'dare_ties': 'dare_ties',
    'regmean': 'regmean',
    'fisher': 'fisher',
    'breadcrumbs': 'breadcrumbs',
    'model_stock': 'model_stock',
    'lamp_merge': 'lamp_merge',
    'lamp-merge': 'lamp_merge',
    'lampmerge': 'lamp_merge',
    'lamp_merge_analysis': 'lamp_merge_analysis',
    'lamp-merge-analysis': 'lamp_merge_analysis',
    'lamp_analysis': 'lamp_merge_analysis',
    'adamerging': 'adamerging',
    'from': 'from',
    'free_merge': 'free_merge',
    'free': 'free_merge',
    'robustmerge': 'robustmerge',
    'robust_merge': 'robustmerge',
    'iso_c': 'iso_c',
    'iso_cts': 'iso_cts',
}


def normalize_method_name(name: str) -> str:
    key = name.strip().lower()
    if key not in METHOD_ALIASES:
        raise ValueError(f'Unsupported merge method: {name}')
    return METHOD_ALIASES[key]


__all__ = [
    'merge_avg',
    'merge_ties',
    'merge_dare_linear',
    'merge_dare_ties',
    'merge_regmean',
    'merge_fisher',
    'merge_breadcrumbs',
    'merge_model_stock',
    'merge_lamp_merge',
    'merge_lamp_merge_analysis',
    'merge_adamerging',
    'merge_from',
    'merge_free',
    'merge_robustmerge',
    'merge_iso_c',
    'merge_iso_cts',
    'normalize_method_name',
]
