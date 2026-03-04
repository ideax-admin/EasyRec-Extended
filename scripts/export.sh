#!/bin/bash
# Export a trained model to SavedModel format
python -c "
from offline.training import OfflineTrainer
trainer = OfflineTrainer(pipeline_config_path='${1:-config/pipeline.config}')
result = trainer.export_model(model_dir='${2:-experiments/model}', export_dir='${3:-exports/latest}')
print(result)
"
