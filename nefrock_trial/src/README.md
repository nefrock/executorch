# executorch

## How to Use

Move this directory's files (executorch/nefrock_trial/src/*.sh) and directory (executorch/nefrock_trial/src/.devcontainer) to the directory where executorch repo is located.  
Then, run the following commands under its directory.

Without QNN
```
source setup.sh
./download.sh
./build.sh
```

With QNN
```
source setup.sh -qnn
./download.sh
./build.sh
```
