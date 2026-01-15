SCRIPT_FILE=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT_FILE)

thirdparty_dir="$SCRIPT_DIR"/../thirdparty
mkdir -p "$thirdparty_dir"

#<<
thirdparty_dir="$SCRIPT_DIR"/../..
#>>

dialect_dir="$SCRIPT_DIR"/../dialects
mkdir -p "$dialect_dir"

triton_repo="$thirdparty_dir"/triton
llvm_repo="$thirdparty_dir"/llvm-project

if [[ ! -e "$triton_repo" ]]; then
    git clone https://github.com/triton-lang/triton.git $triton_repo
fi
if [[ ! -e "$llvm_repo" ]]; then
    git clone --depth 1 --single-branch --branch main https://github.com/llvm/llvm-project.git $llvm_repo
fi

#apt install llvm-17
#apt install mlir-17-dev
#apt install mlir-17-tools

triton_td_files=("TritonAttrDefs.td" "TritonDialect.td" "TritonInterfaces.td" "TritonOpInterfaces.td" "TritonOps.td" "TritonTypes.td")

# Copy each file to destination
for file in "${triton_td_files[@]}"; do
    echo "------ 1"
    if [[ -e "$triton_repo"/include/triton/Dialect/Triton/IR/"$file" ]]; then
         echo "---------2"
         #llvm-tblgen-17 -D ttir "$triton_repo"/include/triton/Dialect/Triton/IR/"$file" -I "$triton_repo"/include/ -I  "$llvm_repo"/mlir/include --dump-json > "$dialect_dir"/"$file".json
	 xdsl-tblgen -i "$dialect_dir"/"$file".json -o "$dialect_dir"/"$file".py 
    else
        echo "Warning: $file does not exist." >&2
    fi
done

