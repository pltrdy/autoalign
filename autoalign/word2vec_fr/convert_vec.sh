#!/bin/bash
set -e
convertvec_dir="./convertvec"
convertvec_bin="$convertvec_dir/convertvec"

get_convertvec(){
    # from http://fauconnier.github.io/
    echo "Running: git clone"
    git clone https://github.com/marekrei/convertvec "$convertvec_dir"
    root="${pwd}"
    cd convertvec/
    echo "Running: make"
    make
    cd "$root"
    
    echo "Done"
}
install(){
    get_convertvec
}
convert(){
    in="$1"
    out="$2"
    echo "Convert vec: '$in' to '$out'"
    cmd="$convertvec_bin bin2txt $in $out"

    echo "Convert cmd: "
    echo "$cmd"

    eval "$cmd"

    echo "Convert done"
}

if [ -z "$1" ]; then
  echo "No action selected"
  echo "Usage: ./convert_vec.sh [install] [convert]"
fi

for action in "$@"; do  
  printf "\n****\nRunning command '$action'\n\n"
  eval "$action" "${@:2}"
  break
done

echo "***"
echo "Done"
