ssh ubuntu@3.71.182.173 sudo apt update
ssh ubuntu@3.71.182.173 sudo apt install python3-pip
ssh ubuntu@3.71.182.173 pip3 install torch torchvision torchaudio
ssh ubuntu@3.71.182.173 "cd ~ && git clone https://github.com/AlexisMotet/projet_3A.git"
ssh ubuntu@3.71.182.173 "cd ~/projet_3A && git switch nextnext"
scp -r C:\Users\alexi\PROJET_3A\Projet_Adversarial_Patch\Project_Adverserial_Patch\Collision_Avoidance\dataset ubuntu@3.71.182.173://home/ubuntu/
scp -r C:\Users\alexi\PROJET_3A\Projet_Adversarial_Patch\Project_Adverserial_Patch\Collision_Avoidance\best_model_extended.pth ubuntu@3.71.182.173://home/ubuntu/model.pth
ssh ubuntu@3.71.182.173 "cd ~/projet_3A && python3 new_generate_adversarial_patch.py"
PAUSE