#!/bin/bash
#PBS -N Spherical_maps_Spair
#PBS -S /bin/bash
#PBS -l hostlist=^shuri,nodes=1:ppn=16:gpus=1,mem=20gb,walltime=24:00:00
#PBS -q student
#PBS -m a
#PBS -M srinivap@informatik.uni-freiburg.de
#PBS -j oe

# For interactive jobs: #PBS -I
# For array jobs: #PBS -t START-END[%SIMULTANEOUS]

echo $(curl google.com)

CUDA_HOME=/misc/software/cuda/cuda-11.7
# PATH=${CUDA_HOME}/bin:${PATH}
# LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# export PATH
# export LD_LIBRARY_PATH
export CUDA_HOME

# echo PATH=${PATH}
# echo LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
echo CUDA_HOME=${CUDA_HOME}

# Setup Repository
if [[ -d "/misc/student/srinivap/git_repo/SphericalMaps" ]]; then
    echo "SphericalMaps is already cloned to /misc/student/srinivap/git_repo/SphericalMaps."
else
    git clone git@github.com:prem-16/SphericalMaps.git /misc/student/srinivap/git_repo/SphericalMaps
fi

while [[ -e "/misc/student/srinivap/git_repo/od3d/installing.txt" ]]; do
    sleep 3
    echo "waiting for installing.txt file to disappear."
done

touch "/misc/student/srinivap/git_repo/od3d/installing.txt"

cd /misc/student/srinivap/git_repo/SphericalMaps


git fetch
git checkout main
git pull



            

# Install OD3D in venv
VENV_NAME=venv_od3d
export VENV_NAME
if [[ -d "${VENV_NAME}" ]]; then
    echo "Venv already exists at /misc/student/srinivap/git_repo/od3d/${VENV_NAME}."
    source /misc/student/srinivap/git_repo/od3d/${VENV_NAME}/bin/activate
else
    echo "Creating venv at /misc/student/srinivap/git_repo/od3d/${VENV_NAME}."
    python3 -m venv /misc/student/srinivap/git_repo/od3d/${VENV_NAME}
    source /misc/student/srinivap/git_repo/od3d/${VENV_NAME}/bin/activate
fi


pip install pip --upgrade
pip install wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -e /misc/student/srinivap/git_repo/od3d
            

rm "/misc/student/srinivap/git_repo/od3d/installing.txt"

DATASET=Animal3D
python3 train_sph.py --config configs/dataset/${DATASET}.yaml


#PYTHONUNBUFFERED=1
#CUDA_VISIBLE_DEVICES=1

exit 0
        