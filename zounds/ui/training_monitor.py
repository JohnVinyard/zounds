from io import BytesIO

from api import ZoundsApp
import tornado.websocket
import tornado.web
import json
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class TrainingMonitorApp(ZoundsApp):
    def __init__(
            self,
            trainer,
            keys_to_graph,
            batch_frequency=10,
            n_training_points=100,
            epoch_key='epoch',
            batch_key='batch',
            base_path=r'/zounds/',
            model=None,
            visualization_feature=None,
            audio_feature=None,
            globals={},
            locals={},
            secret=None):

        super(TrainingMonitorApp, self).__init__(
            base_path=base_path,
            model=model,
            visualization_feature=visualization_feature,
            audio_feature=audio_feature,
            globals=globals,
            locals=locals,
            html='training_monitor.html',
            secret=secret)

        self.n_training_points = n_training_points
        self.batch_frequency = batch_frequency
        self.batch_key = batch_key
        self.epoch_key = epoch_key
        self.keys_to_graph = keys_to_graph
        self.trainer = trainer
        self.training_history = defaultdict(list)
        self.trainer.register_batch_complete_callback(
            self._collect_training_history)

    def custom_routes(self):
        routes = super(TrainingMonitorApp, self).custom_routes()
        routes.extend([
            (r'/zounds/training/?', self.training_handler()),
            (r'/zounds/graph/?', self.graph_handler())
        ])
        return routes

    def _collect_training_history(self, *args, **kwargs):
        batch = kwargs['batch']

        if batch % self.batch_frequency:
            return

        for k in self.keys_to_graph:
            # truncate
            self.training_history[k] = \
                self.training_history[k][-self.n_training_points:]
            # append the new data
            self.training_history[k].append(kwargs[k])

    def training_handler(self):
        app = self

        class TrainingHandler(tornado.websocket.WebSocketHandler):
            def _send_message(self):
                def x(*args, **kwargs):
                    batch = kwargs['batch']

                    if batch % app.batch_frequency:
                        return

                    data = dict(epoch=kwargs['epoch'], batch=batch)
                    for key in app.keys_to_graph:
                        data[key] = kwargs[key]
                    self.write_message(json.dumps(data))

                return x

            def open(self):
                self.func = self._send_message()
                app.trainer \
                    .register_batch_complete_callback(self.func)

            def on_close(self):
                app.trainer \
                    .unregister_batch_complete_callback(self.func)

        return TrainingHandler

    def graph_handler(self):
        app = self

        class GraphHandler(tornado.web.RequestHandler):
            def get(self):
                plt.style.use('dark_background')

                fig = plt.figure()
                handles = []
                for k in app.keys_to_graph:
                    handle, = plt.plot(app.training_history[k], label=k)
                    handles.append(handle)
                plt.legend(handles=handles)

                bio = BytesIO()
                plt.savefig(
                    bio, bbox_inches='tight', pad_inches=0, format='png')
                bio.seek(0)
                fig.clf()
                plt.close('all')
                self.set_header('Content-Type', 'image/png')
                self.write(bio.read())
                self.finish()

        return GraphHandler


class SupervisedTrainingMonitorApp(TrainingMonitorApp):
    def __init__(
            self,
            trainer,
            batch_frequency=10,
            n_training_points=100,
            epoch_key='epoch',
            batch_key='batch',
            base_path=r'/zounds',
            model=None,
            visualization_feature=None,
            audio_feature=None,
            globals={},
            locals={},
            secret=None):
        super(SupervisedTrainingMonitorApp, self).__init__(
            trainer=trainer,
            keys_to_graph=('train_error', 'test_error'),
            model=model,
            batch_frequency=batch_frequency,
            n_training_points=n_training_points,
            epoch_key=epoch_key,
            batch_key=batch_key,
            base_path=base_path,
            visualization_feature=visualization_feature,
            audio_feature=audio_feature,
            globals=globals,
            locals=locals,
            secret=secret)


class GanTrainingMonitorApp(TrainingMonitorApp):
    def __init__(
            self,
            trainer,
            batch_frequency=10,
            n_training_points=100,
            epoch_key='epoch',
            batch_key='batch',
            base_path=r'/zounds',
            model=None,
            visualization_feature=None,
            audio_feature=None,
            globals={},
            locals={},
            secret=None):
        super(GanTrainingMonitorApp, self).__init__(
            trainer=trainer,
            keys_to_graph=('generator_score', 'real_score', 'critic_loss'),
            model=model,
            batch_frequency=batch_frequency,
            n_training_points=n_training_points,
            epoch_key=epoch_key,
            batch_key=batch_key,
            base_path=base_path,
            visualization_feature=visualization_feature,
            audio_feature=audio_feature,
            globals=globals,
            locals=locals,
            secret=secret)


class TripletEmbeddingMonitorApp(TrainingMonitorApp):
    def __init__(
            self,
            trainer,
            batch_frequency=10,
            n_training_points=100,
            epoch_key='epoch',
            batch_key='batch',
            base_path=r'/zounds',
            model=None,
            visualization_feature=None,
            audio_feature=None,
            globals={},
            locals={},
            secret=None):
        super(TripletEmbeddingMonitorApp, self).__init__(
            trainer=trainer,
            keys_to_graph=('error',),
            model=model,
            batch_frequency=batch_frequency,
            n_training_points=n_training_points,
            epoch_key=epoch_key,
            batch_key=batch_key,
            base_path=base_path,
            visualization_feature=visualization_feature,
            audio_feature=audio_feature,
            globals=globals,
            locals=locals,
            secret=secret)
