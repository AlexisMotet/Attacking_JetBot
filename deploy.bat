ssh ubuntu@addr_ip sudo apt update
ssh ubuntu@addr_ip sudo apt upgrade
ssh ubuntu@addr_ip sudo apt install python3-pip
ssh ubuntu@addr_ip pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
ssh ubuntu@addr_ip "cd ~ && git clone https://github.com/AlexisMotet/projet_3A.git"
ssh ubuntu@addr_ip "cd ~/projet_3A && git switch nextnext"
scp -r C:\Users\alexi\PROJET_3A\Projet_Adversarial_Patch\Project_Adverserial_Patch\Collision_Avoidance\dataset ubuntu@addr_ip://home/ubuntu/
scp -r C:\Users\alexi\PROJET_3A\Projet_Adversarial_Patch\Project_Adverserial_Patch\Collision_Avoidance\best_model_extended.pth ubuntu@addr_ip://home/ubuntu/model.pth
ssh ubuntu@addr_ip "cd ~/projet_3A && python3 new_generate_adversarial_patch.py"
PAUSE
