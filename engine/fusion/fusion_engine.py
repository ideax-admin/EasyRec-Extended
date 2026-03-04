class FusionEngine:
    """Engine for fusing multiple recall paths."""
    
    def __init__(self, strategy='weighted'):
        self.strategy = strategy
        self.weights = {
            'popular': 0.3,
            'collaborative': 0.3,
            'content': 0.2,
            'user_based': 0.2
        }
    
    def fuse(self, recall_results):
        """Fuse results from multiple recall paths."""
        if self.strategy == 'weighted':
            return self._weighted_fusion(recall_results)
        elif self.strategy == 'cascade':
            return self._cascade_fusion(recall_results)
        elif self.strategy == 'stacking':
            return self._stacking_fusion(recall_results)
        else:
            return self._weighted_fusion(recall_results)
    
    def _weighted_fusion(self, recall_results):
        """Fuse using weighted average."""
        fused = {}
        for source, items in recall_results.items():
            weight = self.weights.get(source, 0.25)
            for item in items:
                item_id = item.get('item_id')
                if item_id not in fused:
                    fused[item_id] = item.copy()
                    fused[item_id]['score'] = 0
                fused[item_id]['score'] += item.get('score', 0) * weight
        
        return sorted(fused.values(), key=lambda x: x['score'], reverse=True)
    
    def _cascade_fusion(self, recall_results):
        """Cascade fusion - use first non-empty source."""
        for source in ['collaborative', 'popular', 'content', 'user_based']:
            if source in recall_results and recall_results[source]:
                return recall_results[source]
        return []
    
    def _stacking_fusion(self, recall_results):
        """Stack all results together."""
        fused = []
        for source, items in recall_results.items():
            fused.extend(items)
        return fused