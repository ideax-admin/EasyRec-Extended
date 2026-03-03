#!/usr/bin/env python3
"""
EasyRec-Extended 项目初始化脚本
"""

import os
import json
from pathlib import Path

# 项目结构定义
PROJECT_STRUCTURE = {
    'easyrec_extended': {
        '__init__.py': '"""EasyRec Extended Framework"""\n',
        'version.py': '__version__ = "0.1.0"\n',
        'config': {
            '__init__.py': '',
            'policy_config.proto': '// Policy configuration proto definition\n',
            'policy_loader.py': 'from typing import Dict, Any\nimport json\n\nclass PolicyLoader:\n    """Policy 配置加载器"""\n    pass\n',
            'validator.py': '# Validator for policy configuration\n'
        },
        'pipeline': {
            '__init__.py': '',
            'base.py': '# Base pipeline implementation\n',
            'recommendation_engine.py': 'class RecommendationEngine:\n    """推荐引擎"""\n    pass\n',
            'context.py': '# Context management\n'
        },
        'stages': {
            '__init__.py': '',
            'recall': {
                '__init__.py': '',
                'base.py': '# Base recall class\n',
                'neural_recall.py': '# Neural network based recall\n',
                'hotitem_recall.py': '# Hot item recall\n',
                'collaborative_recall.py': '# Collaborative filtering recall\n'
            },
            'fusion': {
                '__init__.py': '',
                'base.py': '# Base fusion strategy\n',
                'merge_dedup.py': '# Merge and dedup fusion\n',
                'weighted_blend.py': '# Weighted blend fusion\n'
            },
            'ranking': {
                '__init__.py': '',
                'easyrec_ranker.py': '# EasyRec ranking integration\n',
                'mtl_ranker.py': '# Multi-task learning ranker\n'
            },
            'post_processing': {
                '__init__.py': '',
                'business_rules.py': '# Business rules\n',
                'diversity.py': '# Diversity control\n',
                'degradation.py': '# Degradation strategy\n'
            }
        },
        'models': {
            '__init__.py': '',
            'model_server.py': '# Model server\n',
            'model_cache.py': '# Model cache\n'
        },
        'features': {
            '__init__.py': '',
            'feature_generator.py': '# Feature generation\n',
            'feature_store.py': '# Feature store integration\n'
        },
        'utils': {
            '__init__.py': '',
            'logger.py': '# Logging utilities\n',
            'metrics.py': '# Metrics utilities\n',
            'helpers.py': '# Helper functions\n'
        }
    },
    'serving': {
        '__init__.py': '',
        'api.py': '# REST API\n',
        'grpc_service.py': '# gRPC service\n',
        'health_check.py': '# Health check\n'
    },
    'examples': {
        'policy_config.json': json.dumps({
            "version": "1.0",
            "name": "default_policy",
            "recall_policies": [],
            "fusion_policy": {},
            "ranking_policy": {},
            "post_processing": {}
        }, indent=2),
        'simple_example.py': '# Simple usage example\n',
        'advanced_example.py': '# Advanced usage example\n'
    },
    'tests': {
        '__init__.py': '',
        'test_config.py': '# Config tests\n',
        'test_pipeline.py': '# Pipeline tests\n',
        'test_stages.py': '# Stages tests\n',
        'test_integration.py': '# Integration tests\n'
    },
    'docs': {
        'README.md': '# Documentation\n',
        'architecture.md': '# Architecture\n',
        'policy_guide.md': '# Policy Configuration Guide\n',
        'deployment.md': '# Deployment Guide\n',
        'api_reference.md': '# API Reference\n'
    },
    'docker': {
        'Dockerfile': 'FROM python:3.9\n',
        'docker-compose.yml': 'version: "3.8"\n',
        '.dockerignore': '__pycache__/\n*.pyc\n.git\n'
    },
    'scripts': {
        'setup.sh': '#!/bin/bash\necho "Setup script"\n',
        'train_model.sh': '#!/bin/bash\necho "Train model script"\n',
        'export_model.sh': '#!/bin/bash\necho "Export model script"\n',
        'deploy.sh': '#!/bin/bash\necho "Deploy script"\n'
    },
    'config': {
        'logging.yaml': 'version: 1\n',
        'server.yaml': 'host: localhost\nport: 8000\n',
        'development.yaml': 'debug: true\n'
    }
}

def create_structure(base_path: str, structure: dict, parent_path: str = ''):
    """递归创建项目结构"""
    created_files = []

    for name, content in structure.items():
        path = os.path.join(parent_path, name)
        full_path = os.path.join(base_path, path)

        if isinstance(content, dict):
            # 创建目录
            os.makedirs(full_path, exist_ok=True)
            # 递归创建子结构
            created_files.extend(create_structure(base_path, content, path))
        else:
            # 创建文件
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            created_files.append(path)
            print(f"✓ Created: {path}")

    return created_files

def main():
    base_path = '.'

    print("🚀 Initializing EasyRec-Extended project structure...\n")
    created_files = create_structure(base_path, PROJECT_STRUCTURE)

    print(f"\n✓ Project structure initialized successfully!")
    print(f"✓ Created {len(created_files)} files\n")

    print("📝 Next steps:")
    print("1. Add and commit all files:")
    print("   git add .")
    print('   git commit -m "Initial commit: Project structure"')
    print("2. Push to GitHub:")
    print("   git push -u origin main")
    print("\n📚 Then run additional setup:")
    print("   pip install -r requirements.txt")
    print("   python setup.py develop")

if __name__ == '__main__':
    main()
