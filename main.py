import warnings, os
warnings.filterwarnings("ignore")
from exp.exp_CSUEI import main


if __name__ == '__main__':
    MAIN_PATH = os.getcwd()
    main(MAIN_PATH)
    # torchrun --nproc_per_node=3 --master_port=29500 main.py




# def set_agent_parameters(args):
#     agent = None
#     device = args.device
#     if args.agent == 'ppo':
#         from agents.ppo.ppo import PPO
#         from agents.ppo.parameters import set_args
#         args.aux_mode = 'off'
#         args.use_planning = 'no'
#         args = set_args(args)
#         args = setup_folders(args)
#         weights = MAIN_PATH+'/results/ppo_lstm_12_testing/checkpoints/'
#         #agent = PPO(args, device, True, weights+'episode_379_Actor.pth', weights+'episode_379_Critic.pth')
#         agent = PPO(args, device, False, '', '')
#         dst = MAIN_PATH + '/results/' + args.folder_id + '/code'
#         copy_folder(MAIN_PATH + '/agents/ppo', dst)