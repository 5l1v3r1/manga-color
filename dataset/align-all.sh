#!/bin/bash

IN_FOLDER=/media/gin/data/share
OUT_FOLDER=/media/gin/data/manga-dataset

echo Naruto
python align_images.py --output_dir $OUT_FOLDER/naruto \
    --input_dir_colored_pattern $IN_FOLDER'/Naruto - Digital Colored Comics/Naruto - Digital Colored Comics * Ch.%s*' \
    --input_dir_gray_pattern $IN_FOLDER'/Naruto/Naruto %s' \
    --start_from 1 --end_at 41 --divide_colored 0

echo Bleach
python align_images.py --output_dir $OUT_FOLDER/bleach \
    --input_dir_colored_pattern $IN_FOLDER'/Bleach - Digital Colored Comics/Bleach - Digital Colored Comics * Ch.%s*' \
    --input_dir_gray_pattern $IN_FOLDER'/Bleach/Bleach %s'  \
    --start_from 350 --end_at 352 --divide_colored 0

echo DeathNote
python align_images.py --output_dir $OUT_FOLDER/death-note \
    --input_dir_colored_pattern $IN_FOLDER'/Death Note Colored Edition/Death Note Colored Edition * Ch.%s' \
    --input_dir_gray_pattern $IN_FOLDER'/Death Note/Death Note Chapter %s'  \
    --start_from 1 --end_at 7 --divide_colored 1

echo ToLoveRu
python align_images.py --output_dir $OUT_FOLDER/love-ru \
    --input_dir_colored_pattern $IN_FOLDER'/To Love-Ru Darkness - Digital Colored Comics/%s' \
    --input_dir_gray_pattern $IN_FOLDER'/To LOVE-RU Darkness/To LOVE-RU Darkness %s *'  \
    --start_from 1 --end_at 31 --divide_colored 0

echo OnePiece
python align_images.py --output_dir $OUT_FOLDER/one-piece \
    --input_dir_colored_pattern $IN_FOLDER'/One Piece - Digital Colored Comics/One Piece - Digital Colored Comics Chapter %s' \
    --input_dir_gray_pattern $IN_FOLDER'/One Piece/One Piece - %s'  \
    --start_from 1 --end_at 721 --divide_colored 0

