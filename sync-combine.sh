rsync --progress -a tomintoul:/disk/scratch/lukedarlow/learning-augmentations/log-combine-images/* log-combine-images/ --exclude=*tar
rsync --progress -a tullibardine:/disk/scratch/lukedarlow/learning-augmentations/log-combine-images/* log-combine-images/ --exclude=*tar
rsync --progress -a wolfburn:/disk/scratch_big/lukedarlow/learning-augmentations/log-combine-images/* log-combine-images/ --exclude=*tar


rsync --progress -a tullibardine:/disk/scratch/lukedarlow/template/log/baseline log-combine-images/ --exclude=*tar
