use std::path::{Path, PathBuf};
use std::sync::Arc;

use itertools::Itertools;
use parking_lot::Mutex;
use segment::common::operation_time_statistics::OperationDurationsAggregator;
use segment::entry::SegmentEntry;
use segment::types::{HnswConfig, HnswGlobalConfig, QuantizationConfig, SegmentType};

use crate::collection_manager::holders::segment_holder::SegmentId;
use crate::collection_manager::optimizers::segment_optimizer::{
    OptimizationPlanner, OptimizerThresholds, SegmentOptimizer,
};
use crate::config::CollectionParams;

const BYTES_IN_KB: usize = 1024;

/// Optimizer that tries to reduce number of segments until it fits configured value.
///
/// It merges 3 smallest segments into a single large segment.
/// Merging 3 segments instead of 2 guarantees that after the optimization the number of segments
/// will be less than before.
pub struct MergeOptimizer {
    default_segments_number: usize,
    thresholds_config: OptimizerThresholds,
    segments_path: PathBuf,
    collection_temp_dir: PathBuf,
    collection_params: CollectionParams,
    hnsw_config: HnswConfig,
    hnsw_global_config: HnswGlobalConfig,
    quantization_config: Option<QuantizationConfig>,
    telemetry_durations_aggregator: Arc<Mutex<OperationDurationsAggregator>>,
}

impl MergeOptimizer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        default_segments_number: usize,
        thresholds_config: OptimizerThresholds,
        segments_path: PathBuf,
        collection_temp_dir: PathBuf,
        collection_params: CollectionParams,
        hnsw_config: HnswConfig,
        hnsw_global_config: HnswGlobalConfig,
        quantization_config: Option<QuantizationConfig>,
    ) -> Self {
        MergeOptimizer {
            default_segments_number,
            thresholds_config,
            segments_path,
            collection_temp_dir,
            collection_params,
            hnsw_config,
            hnsw_global_config,
            quantization_config,
            telemetry_durations_aggregator: OperationDurationsAggregator::new(),
        }
    }
}

impl SegmentOptimizer for MergeOptimizer {
    fn name(&self) -> &str {
        "merge"
    }

    fn segments_path(&self) -> &Path {
        self.segments_path.as_path()
    }

    fn temp_path(&self) -> &Path {
        self.collection_temp_dir.as_path()
    }

    fn collection_params(&self) -> CollectionParams {
        self.collection_params.clone()
    }

    fn hnsw_config(&self) -> &HnswConfig {
        &self.hnsw_config
    }

    fn hnsw_global_config(&self) -> &HnswGlobalConfig {
        &self.hnsw_global_config
    }

    fn quantization_config(&self) -> Option<QuantizationConfig> {
        self.quantization_config.clone()
    }

    fn threshold_config(&self) -> &OptimizerThresholds {
        &self.thresholds_config
    }

    fn plan_optimizations(&self, planner: &mut OptimizationPlanner) {
        let mut candidates = planner
            .remaining()
            .iter()
            .filter_map(|(&segment_id, segment)| {
                let read_segment = segment.read();
                (read_segment.segment_type() != SegmentType::Special).then_some((
                    segment_id,
                    read_segment
                        .max_available_vectors_size_in_bytes()
                        .unwrap_or_default(),
                ))
            })
            .collect_vec();

        // FIXME: in following condition use segment count from planner
        if candidates.len() <= self.default_segments_number {
            return;
        }
        let max_candidates = candidates.len() - self.default_segments_number + 2;

        candidates.sort_by_key(|(_segment_id, size)| *size);
        candidates.truncate(max_candidates);

        // Find at least top-3 smallest segments to join.
        // We need 3 segments because in this case we can guarantee that total segments number will be less
        const MIN_MERGE_SEGMENTS: usize = 3;

        let threshold = self
            .thresholds_config
            .max_segment_size_kb
            .saturating_mul(BYTES_IN_KB);

        let mut candidates: &[(SegmentId, usize)] = &candidates;
        while let batch = take_batch(&mut candidates, threshold)
            && batch.len() >= MIN_MERGE_SEGMENTS
        {
            planner.plan(batch.iter().map(|(segment_id, _)| *segment_id).collect());
        }
    }

    fn get_telemetry_counter(&self) -> &Mutex<OperationDurationsAggregator> {
        &self.telemetry_durations_aggregator
    }
}

/// Takes first N items such that their cumulative size is less than max_size.
/// Advances the input slice forward and returns the taken batch.
fn take_batch<'a, Id: Copy>(items: &mut &'a [(Id, usize)], max_size: usize) -> &'a [(Id, usize)] {
    let it = items.iter().scan(0, |size_sum, &(_segment_id, size)| {
        *size_sum += size;
        (*size_sum < max_size).then_some(())
    });
    let (batch, rest) = items.split_at(it.count());
    *items = rest;
    batch
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use common::budget::ResourceBudget;
    use common::progress_tracker::ProgressTracker;
    use parking_lot::RwLock;
    use segment::index::hnsw_index::num_rayon_threads;
    use tempfile::Builder;

    use super::*;
    use crate::collection_manager::fixtures::{get_merge_optimizer, random_segment};
    use crate::collection_manager::holders::segment_holder::{LockedSegment, SegmentHolder};

    #[test]
    fn test_max_merge_size() {
        let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
        let temp_dir = Builder::new().prefix("segment_temp_dir").tempdir().unwrap();

        let mut holder = SegmentHolder::default();
        let dim = 256;

        let _segments_to_merge = [
            holder.add_new(random_segment(dir.path(), 100, 40, dim)),
            holder.add_new(random_segment(dir.path(), 100, 50, dim)),
            holder.add_new(random_segment(dir.path(), 100, 60, dim)),
        ];

        let mut merge_optimizer = get_merge_optimizer(dir.path(), temp_dir.path(), dim, None);

        let locked_holder = Arc::new(RwLock::new(holder));

        merge_optimizer.default_segments_number = 1;

        merge_optimizer.thresholds_config.max_segment_size_kb = 100;

        let check_result_empty = merge_optimizer.plan_optimizations_for_test(&locked_holder);

        assert!(check_result_empty.is_empty());

        merge_optimizer.thresholds_config.max_segment_size_kb = 200;

        let check_result = merge_optimizer.plan_optimizations_for_test(&locked_holder);
        let check_result = check_result.into_iter().exactly_one().unwrap();

        assert_eq!(check_result.len(), 3);
    }

    #[test]
    fn test_merge_optimizer() {
        let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();
        let temp_dir = Builder::new().prefix("segment_temp_dir").tempdir().unwrap();

        let mut holder = SegmentHolder::default();
        let dim = 256;

        let segments_to_merge = [
            holder.add_new(random_segment(dir.path(), 100, 3, dim)),
            holder.add_new(random_segment(dir.path(), 100, 3, dim)),
            holder.add_new(random_segment(dir.path(), 100, 3, dim)),
            holder.add_new(random_segment(dir.path(), 100, 10, dim)),
        ];

        let other_segment_ids = [
            holder.add_new(random_segment(dir.path(), 100, 20, dim)),
            holder.add_new(random_segment(dir.path(), 100, 20, dim)),
            holder.add_new(random_segment(dir.path(), 100, 20, dim)),
        ];

        let merge_optimizer = get_merge_optimizer(dir.path(), temp_dir.path(), dim, None);

        let locked_holder: Arc<RwLock<_>> = Arc::new(RwLock::new(holder));

        let suggested_to_merge = merge_optimizer.plan_optimizations_for_test(&locked_holder);
        let suggested_for_merge = suggested_to_merge.into_iter().exactly_one().unwrap();
        assert_eq!(suggested_for_merge.len(), 4);

        for segment_in in &suggested_for_merge {
            assert!(segments_to_merge.contains(segment_in));
        }

        let old_path = segments_to_merge
            .iter()
            .map(|sid| match locked_holder.read().get(*sid).unwrap() {
                LockedSegment::Original(x) => x.read().current_path.clone(),
                LockedSegment::Proxy(_) => panic!("Not expected"),
            })
            .collect_vec();

        let permit_cpu_count = num_rayon_threads(0);
        let budget = ResourceBudget::new(permit_cpu_count, permit_cpu_count);
        let permit = budget.try_acquire(0, permit_cpu_count).unwrap();

        merge_optimizer
            .optimize(
                locked_holder.clone(),
                suggested_for_merge,
                permit,
                budget,
                &AtomicBool::new(false),
                ProgressTracker::new_for_test(),
            )
            .unwrap();

        let after_optimization_segments = locked_holder.read().iter().map(|(x, _)| x).collect_vec();

        // Check proper number of segments after optimization
        assert!(after_optimization_segments.len() <= 5);
        assert!(after_optimization_segments.len() > 3);

        // Check other segments are untouched
        for segment_id in &other_segment_ids {
            assert!(after_optimization_segments.contains(segment_id))
        }

        // Check new optimized segment have all vectors in it
        for segment_id in after_optimization_segments {
            if !other_segment_ids.contains(&segment_id) {
                let holder_guard = locked_holder.read();
                let new_segment = holder_guard.get(segment_id).unwrap();
                assert_eq!(new_segment.get().read().available_point_count(), 3 * 3 + 10);
            }
        }

        // Check if optimized segments removed from disk
        old_path.into_iter().for_each(|x| assert!(!x.exists()));
    }

    #[test]
    fn test_take_batch() {
        let mut data = &[(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)][..];
        let batch = take_batch(&mut data, 31);
        assert_eq!(batch, &[(1, 10), (2, 20)]); // sum = 30
        let batch = take_batch(&mut data, 31);
        assert_eq!(batch, &[(3, 30)]); // sum = 30
        let batch = take_batch(&mut data, 31);
        assert_eq!(batch, &[]); // sum = 0
    }
}
