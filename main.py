from types import SimpleNamespace

import os
import json
import time, datetime
import numpy as np

from multiprocessing import Process

import pandas as pd
from tqdm import tqdm

from steps.segmentation.evaluation.evaluate_segmentations import EvaluateSegmentations
from steps.segmentation.evaluation.segmentation_report import SegmentationReport
from steps.segmentation.processing.map_labels import MapLabels
from steps.segmentation.processing.query_sampling import QuerySampling


def setup_least_used_gpu():
    import pynvml

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    min_used_mem = float("inf")
    selected_gpu = 0

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mem = meminfo.used

        print(f"GPU {i}: Used Memory = {used_mem / 1024**2:.2f} MB")
        if used_mem < min_used_mem:
            min_used_mem = used_mem
            selected_gpu = i

    pynvml.nvmlShutdown()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    print(f"Selected GPU: {selected_gpu}")
    return selected_gpu


def setup_gpu(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if gpu_id == "auto":
        return setup_least_used_gpu()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Selected GPU from argument: {gpu_id}")

    return gpu_id


def main():
    """
    Usage:
        main.py [options]
        main.py --help | -h

    Options:
        -p, --preprocess                Run preprocessing
        -t, --train                     Run training
        -g, --generate                  Generate segmentation volumes
        -e, --evaluate                  Run evaluation

        --gpu=<id>                      GPU id to use, or auto [default: auto]
        -h, --help                      Show this help
    """

    from docopt import docopt
    arguments = docopt(main.__doc__)

    experiment_name = "dpt_siren_1k_v2"

    # define paths
    project_root = "/path/to/project"
    data_source = "/path/to/data"
    training_patient_list_file = os.path.join(data_source, "training_patient_list.txt")
    test_patient_list_file = os.path.join(data_source, "test_patient_list.txt")
    raw_label_json_path = os.path.join(data_source, "organ_label_list.json")
    train_label_json_path = os.path.join(data_source, "label_organs.json")
    body_data_path = os.path.join(data_source,"data_1k_python37.npz")
    mask_dir = os.path.join(data_source, 'masks_volumetric_preprocessed_v2')
    prefix = os.path.join(project_root, "experiments")
    experiment_root = os.path.join(prefix, experiment_name)
    preprocess_data = os.path.join(experiment_root, "1_preprocess")
    training_data = os.path.join(experiment_root, "2_train")
    generate_data = os.path.join(experiment_root, "3_generate/weights_last")
    evaluate_data = os.path.join(experiment_root, "4_evaluate/weights_last")

    # Load Patient Data
    train_patient_list = []
    with open(training_patient_list_file, "r") as f:
        for line in f:
            train_patient_list.append(line.strip())
    test_patient_list = []
    with open(test_patient_list_file, "r") as f:
        for line in f:
            test_patient_list.append(line.strip())

    # Body Configuration
    cfg_body = SimpleNamespace()
    cfg_body.num_encoder = 6
    cfg_body.planes = [16, 32, 64, 128, 256, 512]
    cfg_body.blocks = [2, 3, 4, 5, 6, 3]
    cfg_body.share_planes = 8
    cfg_body.stride = [1, 4, 4, 4, 4, 4]
    cfg_body.nsample = [8, 8, 16, 16, 16, 16]


    # define training parameters
    batch_size = 8
    n_epoch = 300

    with open(train_label_json_path, "r") as file:
        train_labels = json.load(file)
    num_classes = len(train_labels) + 1

    # define model parameters
    c_dim = cfg_body.planes[-1]
    learning_rate = 0.0001
    weight_decay = 0.0
    momentum = 0.90

    # value mapping for train labels
    with open(raw_label_json_path, "r") as file:
        organ_to_raw_label = json.load(file)
    with open(train_label_json_path, "r") as file:
        train_id_to_organ = json.load(file)
    value_mapping = {0: 0}
    name_mapping = {0: "Background"}

    for train_id, organ in train_id_to_organ.items():
        raw_label = int(organ_to_raw_label[organ])
        train_label = int(train_id) + 1

        value_mapping[raw_label] = train_label
        name_mapping[train_label] = organ

    if arguments["--preprocess"]:
        os.makedirs(preprocess_data, exist_ok=True)
        mapped_mask_dir = os.path.join(preprocess_data, "mapped_masks")

        mapper = MapLabels(
            input_dir=mask_dir,
            output_dir=mapped_mask_dir,
            value_mapping=value_mapping,
            name_mapping=name_mapping
        )

        all_patient_ids = sorted(
            set(train_patient_list) | set(test_patient_list)
        )

        mapper.process(patient_ids=all_patient_ids)


        sampler = QuerySampling(
            mask_dir=mapped_mask_dir,
            train_label_json_path=train_label_json_path,
            n_per_organ=1024,
            n_global_background=4096,
            near_surface_ratio=0.9,
            surface_sigmas=(1.0, 3.0),
            normalize_to_dpt=True,
            log_ratio=False,
        )

        query_samples_root = os.path.join(
            preprocess_data,
            "query_samples",
        )

        train_output_dir = os.path.join(
            query_samples_root,
            "train",
        )

        sampler.sample(
            output_dir=train_output_dir,
            patient_ids=train_patient_list,
        )

    if arguments["--train"]:
        setup_gpu(arguments["--gpu"])

        import torch
        import torch.optim as optim
        from tensorboardX import SummaryWriter

        from data.core import NAKO10KBodyDataset, collate_fn

        from models import BodyImplicitSegmentationNetwork
        from models.encoder import GlobalBodyEncoderLN
        from models.decoder import ModulatedSirenDecoder

        from steps.training.sconet.training import Trainer

        from util.nn.pytorch.losses import CEDiceLoss

        t0 = time.time()

        device = torch.device("cuda:0")
        torch.cuda.set_device(0)

        # Output directory
        os.makedirs(training_data, exist_ok=True)

        # Dataset and loaders
        body_data = np.load(body_data_path, allow_pickle=False)
        train_query_dir = os.path.join(preprocess_data, "query_samples", "train")
        train_dataset = NAKO10KBodyDataset(
            patient_ids=train_patient_list,
            body_data=body_data,
            mode="train",
            query_samples_dir=train_query_dir,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn)


        #Model
        body_encoder = GlobalBodyEncoderLN(cfg_body=cfg_body, c=3)
        decoder = ModulatedSirenDecoder(c_dim=c_dim, hidden_size=256, n_layers=5, num_classes=num_classes, w0_initial=5.)
        model = BodyImplicitSegmentationNetwork(encoder=body_encoder, decoder=decoder, device=device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(momentum, 0.999))

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=device,
            loss=CEDiceLoss(alpha=1.0, beta=1.0, do_bg=False)
        )

        it = 0
        best_train_loss = float("inf")

        logger = SummaryWriter(
            os.path.join(training_data, "logs")
        )

        nparameters = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {nparameters}")

        for epoch_it in range(1, n_epoch + 1):
            epoch_loss_train = []

            for batch in train_loader:
                it += 1
                result = trainer.train_step(batch)
                epoch_loss_train.append(result["loss"])

            avg_loss = sum(epoch_loss_train) / len(epoch_loss_train)

            logger.add_scalar("train/loss", avg_loss, epoch_it)

            if avg_loss <= best_train_loss:
                best_train_loss = avg_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(training_data, "model_weights_best.pth"),
                )

            current_time = datetime.datetime.now()

            print(
                f"[Epoch {epoch_it:03d}/{n_epoch:03d}] "
                f"it={it:06d}, loss={avg_loss:.4f}, "
                f"time={time.time() - t0:.2f}s, "
                f"{current_time.hour:02d}:{current_time.minute:02d}"
            )

        print("Maximum number of epochs reached.")

        torch.save(
            model.state_dict(),
            os.path.join(training_data, "model_weights_last.pth"),
        )

        with open(
                os.path.join(training_data, "training_details.txt"),
                "w",
        ) as file:
            file.write(f"Total number of parameters: {nparameters}\n")
            file.write(f"Epochs: {n_epoch}\n")
            file.write(f"Iterations: {it}\n")
            file.write(f"Final training loss: {avg_loss:.6f}\n")
            file.write(f"Best training loss: {best_train_loss:.6f}\n")

        logger.close()

    if arguments["--generate"]:
        setup_gpu(arguments["--gpu"])

        import torch

        from common.gpu_monitor import daemon_process

        from data.core import NAKO10KBodyDataset, collate_fn

        from models import BodyImplicitSegmentationNetwork
        from models.encoder import GlobalBodyEncoderLN
        from models.decoder import ModulatedSirenDecoder

        from steps.segmentation.processing.segmentation_volume_generator import SegmentationVolumeGenerator

        from util.tools import to_nii

        device = torch.device("cuda:0")
        torch.cuda.set_device(0)

        # Output folder
        volume_dir = os.path.join(generate_data, "volumes")
        os.makedirs(volume_dir, exist_ok=True)
        out_time_file = os.path.join(generate_data, 'perf_generation_full.csv')

        #Dataset
        body_data = np.load(body_data_path, allow_pickle=False)
        metadata_csv = os.path.join(data_source, 'masks_volumetric_metadata.csv')
        test_dataset = NAKO10KBodyDataset(
            patient_ids=test_patient_list,
            body_data=body_data,
            mode="test",
            metadata_csv=metadata_csv,
        )

        #Model
        body_encoder = GlobalBodyEncoderLN(cfg_body=cfg_body, c=3)
        decoder = ModulatedSirenDecoder(c_dim=c_dim, hidden_size=256, n_layers=5, num_classes=num_classes, w0_initial=5.)
        model = BodyImplicitSegmentationNetwork(encoder=body_encoder, decoder=decoder, device=device)


        model_weights_path = os.path.join(
            training_data,
            "model_weights_last.pth",
        )

        model.load_state_dict(
            torch.load(model_weights_path, map_location=device)
        )

        # Generator
        generator = SegmentationVolumeGenerator(
            model=model,
            device=device,
            num_classes=num_classes,
        )

        # Loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn=collate_fn)

        # Statistics
        perf_dicts = []

        # Generate
        model.eval()

        # Check GPU usage during segmentation generation
        gpu_usage_path = os.path.join(generate_data, "gpu_usage_inference.txt")
        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(device=device)
            with open(gpu_usage_path, "w") as f:
                f.write(device_name + "\n")
            p1 = Process(target=daemon_process, args=(1, gpu_usage_path))
            p1.daemon = True
            p1.start()


        for it, data in enumerate(tqdm(test_loader)):
            patient_id = data["patient_ids"][0]

            # Timing dict
            perf_dict = {
                "patient_id": patient_id,
            }

            dim = tuple(
                int(value)
                for value in data["dim"][0].tolist()
            )

            t0 = time.time()
            out = generator.generate_volume(data,
                                            dim=dim)

            perf_dict['inference_time'] = time.time() - t0

            volume = out['volume']

            # Get statistics
            stats_dict = {}
            if 'stats_dict' in out:
                stats_dict = out['stats_dict']

            perf_dict.update(stats_dict)

            #Write output
            volume_out_file = os.path.join(volume_dir, patient_id + ".nii.gz")
            to_nii(
                volume,
                volume_out_file,
                spacing=data["spacing"][0].tolist(),
                origin=data["origin"][0].tolist(),
                direction=data["direction"][0].tolist(),
            )

            if 'flop_counter' in out:
                perf_dict['flops'] = out['flop_counter']

            # save GPU and time usage
            perf_dicts.append(perf_dict)

        #Create pandas dataframe and save
        time_df = pd.DataFrame(perf_dicts)
        time_df.set_index(['patient_id'], inplace=True)
        time_df.to_csv(out_time_file)

    if arguments["--evaluate"]:
        os.makedirs(evaluate_data, exist_ok=True)

        pred_dir = os.path.join(generate_data, "volumes")
        gt_dir = os.path.join(prefix, "masks_volumetric_preprocessed_v2_mapped")

        evaluator = EvaluateSegmentations(
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            patient_ids=test_patient_list,
            name_mapping=name_mapping,
            selected_metrics=['DSC', 'HDD95', 'USD', 'ASSD'],
            metrics_dir=evaluate_data,
        )

        evaluator.eval_all()

        csv_path = os.path.join(evaluate_data, "eval_all.csv")
        output_path = os.path.join(evaluate_data, "analysis/segmentation_report.html")

        report = SegmentationReport(
            csv_path=csv_path,
            output_path=output_path,
        )

        report.create()

if __name__ == "__main__":
    main()







