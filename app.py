import logging
from flask import Flask, jsonify, request
from online.serving import RecommendationServer
from core.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
config = get_config()
server = RecommendationServer(config)

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    result_size = request.args.get('result_size', type=int)
    result = server.get_recommendations(user_id=user_id, result_size=result_size)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify(server.health_check())

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': config.SERVICE_NAME,
        'version': config.VERSION,
        'endpoints': {
            'recommend': '/recommend?user_id=<user_id>',
            'health': '/health'
        }
    })

if __name__ == '__main__':
    logger.info(f'Starting {config.SERVICE_NAME} v{config.VERSION}')
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.DEBUG)