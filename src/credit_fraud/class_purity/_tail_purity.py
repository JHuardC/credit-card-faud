"""
Calculates percent of class within tail of distribution for each
element.
"""
### Imports
from typing import Union, Callable
from operator import lt, gt
from functools import partial
from numpy import unique, number, array
from numpy.typing import NDArray, ArrayLike

### Function
def _class_purity(
    feature: NDArray,
    y: ArrayLike,
    cls_lbl: number,
    op: Callable[[number, number], bool],
    cut_offs: Union[NDArray, None] = None
) -> NDArray:
    """
    Measures a particular class's purity within some range of the
    feature distribution defined by op and cut_offs.

    Parameters
    ----------

    feature: NDArray
        Array of a single feature in the data.

    y: ArrayLike of number.
        Array of labels, one label is selected in cls_lbl to measure
        it's purity against the other labels.

    cls_lbl: number.
        Label value to measure purity.
    
    op: callable.
        Comparison operator used to define the range of the feature
        space within which purity is measured.
    
    cut_offs: NDArray. Optional.
        Key values to apply op to; if none then unique values of the
        feature space are used
    
    Returns
    -------

        2-dimensional array (, 2) of feature values and purity scores.
    """
    cut_offs_is_none = isinstance(cut_offs, type(None))
    unique_vals = unique(cut_offs if not cut_offs_is_none else feature)
    y_is_lbl = y == cls_lbl

    purity = []
    for el in unique_vals:
        
        y_in_tail = y_is_lbl[op(feature, el)]
        
        try:
            purity.append([el, (y_in_tail).sum() / len(y_in_tail)])

        except ZeroDivisionError as e:
            # implies len(y_in_tail) is 0
            purity.append([el, 0])

        except Exception as e:
            raise e # Raise all other exceptions

    return array(purity)


def less_than_tail_class_purity(
    feature: NDArray,
    y: ArrayLike,
    cls_lbl: number,
    cut_offs: Union[NDArray, None] = None
) -> NDArray:
    """
    Measures a particular class's purity within the less than tail of
    the feature distribution.

    Parameters
    ----------

    feature: NDArray
        Array of a single feature in the data.

    y: ArrayLike of number.
        Array of labels, one label is selected in cls_lbl to measure
        it's purity against the other labels.

    cls_lbl: number.
        Label value to measure purity.
    
    cut_offs: NDArray. Optional.
        Key values to apply op to; if none then unique values of the
        feature space are used
    
    Returns
    -------

        2-dimensional array (, 2) of feature values and purity scores.
    """
    return _class_purity(feature, y, cls_lbl, op = lt, cut_offs = cut_offs)


def greater_than_tail_class_purity(
    feature: NDArray,
    y: ArrayLike,
    cls_lbl: number,
    cut_offs: Union[NDArray, None] = None
) -> NDArray:
    """
    Measures a particular class's purity within the greater than tail of
    the feature distribution.

    Parameters
    ----------

    feature: NDArray
        Array of a single feature in the data.

    y: ArrayLike of number.
        Array of labels, one label is selected in cls_lbl to measure
        it's purity against the other labels.

    cls_lbl: number.
        Label value to measure purity.
    
    cut_offs: NDArray. Optional.
        Key values to apply op to; if none then unique values of the
        feature space are used
    
    Returns
    -------

        2-dimensional array (, 2) of feature values and purity scores.
    """
    return _class_purity(feature, y, cls_lbl, op = gt, cut_offs = cut_offs)
