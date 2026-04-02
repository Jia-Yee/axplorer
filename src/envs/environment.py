import statistics
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from logging import getLogger

logger = getLogger()


class DataPoint(ABC):
    def __init__(self):
        super().__init__()
        self.score = -1
        self.features = ""

    @abstractmethod
    def calc_score(self):
        pass

    @abstractmethod
    def calc_features(self):
        pass

    def local_search(self, improve_with_local_search):
        return

    @classmethod
    def _update_class_params(cls, pars):
        return

    @classmethod
    def _batch_generate_and_score(cls, batch_size, N, pars=None):
        out = []
        if pars is not None:
            cls._update_class_params(pars)
        for _ in range(batch_size):
            d = cls(N=N, init=True)
            if d.score >= 0:
                out.append(d)
        return out


class BaseEnvironment(object):
    data_class = None
    SPECIAL_SYMBOLS = ["SEP", "EOS", "PAD", "BOS"]

    def __init__(self, params):
        return


def compute_stats(scores):
    num_bins = 200
    if len(scores) == 0:
        logger.info(f"No valid examples")
        return None

    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    top_1_percentile = statistics.quantiles(scores, n=100)[-1] if len(scores) >= 100 else max(scores)
    max_score = max(scores)

    logger.info(f"Valid examples: {len(scores)}")
    logger.info(f"Mean score: {mean}")
    logger.info(f"Median score: {median}")
    logger.info(f"Stdev score: {stdev}")
    logger.info(f"Max score: {max_score}")
    logger.info(f"Top 1 percentile score: {top_1_percentile}")

    logger.info(f"Distribution of scores:")
    counts = Counter(sorted(scores))
    if len(counts) > num_bins:
        min_score, max_score = min(scores), max(scores)
        bin_width = (max_score - min_score) / num_bins
        bins = Counter()
        for score, count in counts.items():
            bin_idx = min(int((score - min_score) / bin_width), num_bins - 1)
            bin_start = min_score + bin_idx * bin_width
            bin_end = bin_start + bin_width
            bins[(bin_start, bin_end)] += count
        for (start, end), count in bins.items():
            logger.info(f"Score [{start:.2f}, {end:.2f}): Count: {count}")
    else:
        for score, count in counts.items():
            logger.info(f"Score {score}: Count: {count}")
    logger.info("--------------------------------")
    return {"mean": mean, "median": median, "top_1_percentile": top_1_percentile, "max": max_score}


def do_stats(n_invalid, data):
    """
    Compute and log statistics
    """
    scores = [d.score for d in data if d.score >= 0]
    logger.info(f"### Score distribution ###")
    if n_invalid >= 0:
        logger.info(f"Invalid examples: before local search: {n_invalid}, after: {len(data) - len(scores)}")
    return compute_stats(scores)


def _do_score(d, always_search: bool = False, redeem_only: bool = False, pars=None):
    invalid = 0
    if pars is not None:
        d._update_class_params(pars)
    
    # ✅ OPTIMIZED: Compute score first, only compute features for valid samples
    d.calc_score()
    
    # Only compute features if sample is valid AND we need uniqueness checking
    if d.score >= 0:
        if hasattr(d, 'calc_features'):
            d.calc_features()
    else:
        # Invalid samples don't need features (they'll be filtered out anyway)
        d.features = ""
        invalid = 1
    
    # Local search if needed
    if always_search:
        d.local_search(improve_with_local_search=True)
    elif invalid and redeem_only:
        d.local_search(improve_with_local_search=False)
    
    return (d, invalid)


def do_score(data, args, executor=None):
    """
    Compute the score of a list of data.
    Can be parallelized with process_pool.
    Returns only valid items (score >= 0).
    
    Optimized: Automatically use multiprocessing if executor is provided.
    """
    n_invalid = 0
    processed_data = []
    
    # Handle empty data list
    if len(data) == 0:
        return [], 0, []
    
    # ✅ OPTIMIZED: Use executor if available, otherwise sequential
    if executor is not None:
        # Parallel execution with ProcessPoolExecutor
        pars = data[0]._save_class_params()
        chunksize = max(1, len(data) // (executor._max_workers * 32))
        
        for d, invalid in executor.map(_do_score, data, repeat(args.always_search), repeat(args.redeem_only), repeat(pars), chunksize=chunksize):
            processed_data.append(d)
            n_invalid += invalid
    else:
        # Sequential execution (fallback)
        for d in data:
            res, invalid = _do_score(d, args.always_search, args.redeem_only)
            processed_data.append(res)
            n_invalid += invalid

    valid_data = [d for d in processed_data if d.score >= 0]

    return valid_data, n_invalid, processed_data
