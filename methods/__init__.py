from .avg import merge_avg
from .avg_head import merge_avg_head
from .dare import merge_dare_linear, merge_dare_ties
from .adamerging import merge_adamerging
from .breadcrumbs import merge_breadcrumbs
from .fisher import merge_fisher
from .from_merge import merge_from
from .free_merge import merge_free
from .head_only import merge_head_only
from .iso import merge_iso_c, merge_iso_cts
from .model_stock import merge_model_stock
from .lamp_merge import merge_lamp_merge
from .lamp_merge_analysis import merge_lamp_merge as merge_lamp_merge_analysis
from .regmean import merge_regmean
from .robustmerge import merge_robustmerge
from .ties import merge_ties

METHOD_ALIASES = {
    'avg': 'avg',
    'avg_head': 'avg_head',
    'avg-head': 'avg_head',
    'head_avg': 'head_avg',
    'head-avg': 'head_avg',
    'ties': 'ties',
    'head_ties': 'head_ties',
    'head-ties': 'head_ties',
    'dare_linear': 'dare_linear',
    'dare-liner': 'dare_linear',
    'dare_liner': 'dare_linear',
    'head_dare_linear': 'head_dare_linear',
    'head-dare-linear': 'head_dare_linear',
    'dare_ties': 'dare_ties',
    'head_dare_ties': 'head_dare_ties',
    'head-dare-ties': 'head_dare_ties',
    'regmean': 'regmean',
    'head_regmean': 'head_regmean',
    'head-regmean': 'head_regmean',
    'fisher': 'fisher',
    'head_fisher': 'head_fisher',
    'head-fisher': 'head_fisher',
    'breadcrumbs': 'breadcrumbs',
    'head_breadcrumbs': 'head_breadcrumbs',
    'head-breadcrumbs': 'head_breadcrumbs',
    'model_stock': 'model_stock',
    'head_model_stock': 'head_model_stock',
    'head-model-stock': 'head_model_stock',
    'lamp_merge': 'lamp_merge',
    'lamp-merge': 'lamp_merge',
    'lampmerge': 'lamp_merge',
    'lamp_merge_analysis': 'lamp_merge_analysis',
    'lamp-merge-analysis': 'lamp_merge_analysis',
    'lamp_analysis': 'lamp_merge_analysis',
    'adamerging': 'adamerging',
    'from': 'from',
    'head_from': 'head_from',
    'head-from': 'head_from',
    'free_merge': 'free_merge',
    'free': 'free_merge',
    'head_free_merge': 'head_free_merge',
    'head-free-merge': 'head_free_merge',
    'head_free': 'head_free_merge',
    'robustmerge': 'robustmerge',
    'robust_merge': 'robustmerge',
    'head_robustmerge': 'head_robustmerge',
    'head-robustmerge': 'head_robustmerge',
    'head_robust_merge': 'head_robustmerge',
    'iso_c': 'iso_c',
    'head_iso_c': 'head_iso_c',
    'head-iso-c': 'head_iso_c',
    'iso_cts': 'iso_cts',
}


def normalize_method_name(name: str) -> str:
    key = name.strip().lower()
    if key not in METHOD_ALIASES:
        raise ValueError(f'Unsupported merge method: {name}')
    return METHOD_ALIASES[key]


__all__ = [
    'merge_avg',
    'merge_avg_head',
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
    'merge_head_only',
    'merge_robustmerge',
    'merge_iso_c',
    'merge_iso_cts',
    'normalize_method_name',
]
