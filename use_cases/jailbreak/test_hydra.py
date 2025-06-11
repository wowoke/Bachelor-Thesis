import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, 
            config_path="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/src/resp_gen_ai/hydra_files", 
            config_name="config")
def main(cfg: DictConfig):
    print("Experiment Name:", cfg.experiment_name)
    print("Seed:", cfg.seed)
    print("Logging level:", cfg.logging.level)
    print("Target Model:", cfg.model.testing_model)
    print("DB Name:", cfg.data_base.database)
    print("Judge Model:", cfg.judge_model.judge_testing_model)

    print("Results will be saved to:", cfg.logging.save_path)

if __name__ == "__main__":
    main()
