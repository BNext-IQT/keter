from flask import Flask
import rq_dashboard

application = Flask(__name__)
application.config.from_object(rq_dashboard.default_settings)
application.register_blueprint(rq_dashboard.blueprint, url_prefix="/rq")