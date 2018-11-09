rsync --progress -a tomintoul:/disk/scratch/lukedarlow/learning-augmentations/log-masked/* log-masked/ --exclude=*tar
rsync --progress -a tullibardine:/disk/scratch/lukedarlow/learning-augmentations/log-masked/* log-masked/ --exclude=*tar
rsync --progress -a wolfburn:/disk/scratch_big/lukedarlow/learning-augmentations/log-masked/* log-masked/ --exclude=*tar


rsync --progress -a tullibardine:/disk/scratch/lukedarlow/template/log/baseline log-masked/ --exclude=*tar
