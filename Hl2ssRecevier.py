import hl2ss.viewer.hl2ss as hl2ss
import hl2ss.viewer.hl2ss_mp as hl2ss_mp
import hl2ss.viewer.hl2ss_lnm as hl2ss_lnm
import hl2ss.viewer.hl2ss_3dcv as hl2ss_3dcv
import numpy as np
import multiprocessing as mp

class Hl2ssStreamer:
    def __init__(self, host, calibration_path, data_root_path='data', pv_width=640, pv_height=360, pv_fps=30, buffer_size=10):
        self.host = host
        self.calibration_path = calibration_path
        self.data_root_path = data_root_path
        self.pv_width = pv_width
        self.pv_height = pv_height
        self.pv_fps = pv_fps
        self.buffer_size = buffer_size
        
        # Start PV Subsystem
        hl2ss_lnm.start_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)
        
        # Get RM Depth AHAT calibration
        self.calibration_ht = hl2ss_3dcv.get_calibration_rm(self.host, hl2ss.StreamPort.RM_DEPTH_AHAT, self.calibration_path)
        self.uv2xy = self.calibration_ht.uv2xy
        self.xy1, self.scale = hl2ss_3dcv.rm_depth_compute_rays(self.uv2xy, self.calibration_ht.scale)
        self.max_depth = self.calibration_ht.alias / self.calibration_ht.scale

        # Initialize folders to store data
        self.init_storage_folders()

        # Start PV and RM Depth AHAT streams
        self.producer, self.consumer, self.sink_pv, self.sink_ht = self.start_streams()

        # Initialize PV intrinsics and extrinsics
        self.pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
        self.pv_extrinsics = np.eye(4, 4, dtype=np.float32)


    def start_streams(self):
        producer = hl2ss_mp.producer()
        producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO,
                           hl2ss_lnm.rx_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO, width=self.pv_width, height=self.pv_height, framerate=self.pv_fps))
        producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(self.host, hl2ss.StreamPort.RM_DEPTH_AHAT))
        producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, self.pv_fps * self.buffer_size)
        producer.initialize(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss.Parameters_RM_DEPTH_AHAT.FPS * self.buffer_size)
        producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
        producer.start(hl2ss.StreamPort.RM_DEPTH_AHAT)

        consumer = hl2ss_mp.consumer()
        manager = mp.Manager()
        sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
        sink_ht = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_AHAT, manager, None)

        sink_pv.get_attach_response()
        sink_ht.get_attach_response()

        return producer, consumer, sink_pv, sink_ht

    def get_frames(self):
        """
        Ottiene il frame più recente dal sink per la Depth camera e il frame PV
        sincronizzato.
        """
        _, data_ht = self.sink_ht.get_most_recent_frame()
        if ((data_ht is None) or (not hl2ss.is_valid_pose(data_ht.pose))):
            return None, None

        _, data_pv = self.sink_pv.get_nearest(data_ht.timestamp)
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            return None, None

        return data_pv, data_ht

    def get_current_frame(self):
        """
        Restituisce il frame corrente come immagine a colori e mappa di profondità.
        """
        data_pv, data_ht = self.get_frames()
        if data_pv is None or data_ht is None:
            return None, None

        # Preprocessamento e sincronizzazione dei frame
        depth = data_ht.payload.depth
        z = hl2ss_3dcv.rm_depth_normalize(depth, self.scale)
        color = data_pv.payload.image

        # Ridimensiona la mappa di profondità per adattarsi ai valori di profondità massimi
        depth_normalized = (z * 255 / self.max_depth).astype(np.uint8)

        # Ritorna l'immagine RGB e la mappa di profondità
        return color, depth_normalized

    def cleanup(self):
        """
        Ferma i flussi e pulisce le risorse.
        """
        self.sink_pv.detach()
        self.sink_ht.detach()
        self.producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
        self.producer.stop(hl2ss.StreamPort.RM_DEPTH_AHAT)
        hl2ss_lnm.stop_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)

