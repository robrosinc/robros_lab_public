#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-igris_c_isaaclab}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ISAACSIM_DIR="${REPO_ROOT}/third_party/IsaacSim"
ISAACLAB_DIR="${REPO_ROOT}/third_party/IsaacLab"
RSL_RL_DIR="${REPO_ROOT}/third_party/rsl_rl"
ROBROS_LAB_DIR="${REPO_ROOT}/source/robros_lab_public"

case "$(uname -m)" in
    x86_64)
        CONDA_ARCH="x86_64"
        ISAACSIM_BUILD_DIR="${ISAACSIM_DIR}/_build/linux-x86_64/release"
        ;;
    aarch64|arm64)
        CONDA_ARCH="aarch64"
        ISAACSIM_BUILD_DIR="${ISAACSIM_DIR}/_build/linux-aarch64/release"
        ;;
    *)
        echo "Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This setup script only supports Linux." >&2
    exit 1
fi

if [[ $EUID -eq 0 ]]; then
    SUDO=()
else
    SUDO=(sudo)
fi

ensure_apt_packages() {
    if ! command -v apt-get >/dev/null 2>&1; then
        return
    fi

    local packages=()
    local need_update=0

    command -v curl >/dev/null 2>&1 || packages+=(curl)
    command -v git >/dev/null 2>&1 || packages+=(git)
    command -v wget >/dev/null 2>&1 || packages+=(wget)
    command -v bzip2 >/dev/null 2>&1 || packages+=(bzip2)
    command -v cmake >/dev/null 2>&1 || packages+=(cmake)
    command -v make >/dev/null 2>&1 || packages+=(build-essential)
    git lfs version >/dev/null 2>&1 || packages+=(git-lfs)

    if ((${#packages[@]} > 0)); then
        need_update=1
    fi

    if [[ ! -x /usr/bin/gcc-11 || ! -x /usr/bin/g++-11 ]]; then
        need_update=1
    fi

    if ((need_update)); then
        "${SUDO[@]}" apt-get update
    fi

    if ((${#packages[@]} > 0)); then
        "${SUDO[@]}" apt-get install -y --no-install-recommends "${packages[@]}"
    fi

    if [[ ! -x /usr/bin/gcc-11 || ! -x /usr/bin/g++-11 ]]; then
        "${SUDO[@]}" apt-get install -y --no-install-recommends gcc-11 g++-11 || true
    fi
}

ensure_conda() {
    if command -v conda >/dev/null 2>&1; then
        return
    fi

    if [[ -x "${CONDA_ROOT}/bin/conda" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
        return
    fi

    if [[ -e "${CONDA_ROOT}" ]]; then
        echo "Found existing path at ${CONDA_ROOT}, but it does not look like a valid Miniconda install." >&2
        echo "Remove it or set CONDA_ROOT to the correct conda installation path." >&2
        exit 1
    fi

    local installer="Miniconda3-latest-Linux-${CONDA_ARCH}.sh"
    local installer_path="/tmp/${installer}"
    mkdir -p "$(dirname "${CONDA_ROOT}")"
    curl -fsSL "https://repo.anaconda.com/miniconda/${installer}" -o "${installer_path}"
    bash "${installer_path}" -b -p "${CONDA_ROOT}"
    rm -f "${installer_path}"
}

source_conda() {
    local conda_base

    if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
        conda_base="${CONDA_ROOT}"
    elif command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
    else
        echo "Unable to locate conda.sh." >&2
        exit 1
    fi

    # shellcheck disable=SC1090
    source "${conda_base}/etc/profile.d/conda.sh"
}

conda_activate_safe() {
    local env_name="$1"
    local had_nounset=0

    case "$-" in
        *u*) had_nounset=1 ;;
    esac

    set +u
    conda activate "${env_name}"
    if ((had_nounset)); then
        set -u
    fi
}

accept_conda_tos() {
    local old_no_plugins="${CONDA_NO_PLUGINS-}"
    unset CONDA_NO_PLUGINS

    if ! conda tos --help >/dev/null 2>&1; then
        if [[ -n "${old_no_plugins}" ]]; then
            export CONDA_NO_PLUGINS="${old_no_plugins}"
        fi
        return
    fi

    local channels=(
        "https://repo.anaconda.com/pkgs/main"
        "https://repo.anaconda.com/pkgs/r"
    )
    local channel

    for channel in "${channels[@]}"; do
        conda tos accept --override-channels --channel "${channel}" >/dev/null
    done

    if [[ -n "${old_no_plugins}" ]]; then
        export CONDA_NO_PLUGINS="${old_no_plugins}"
    else
        unset CONDA_NO_PLUGINS
    fi
}

ensure_conda_libmamba() {
    local solver=""

    solver="$(conda config --show solver 2>/dev/null | awk '/solver:/ {print $2}' || true)"
    if conda list -n base conda-libmamba-solver >/dev/null 2>&1; then
        conda config --set solver libmamba >/dev/null 2>&1 || true
        return
    fi

    echo "[INFO] Installing conda-libmamba-solver into base..."
    conda install -n base -y conda-libmamba-solver

    echo "[INFO] Switching conda solver to libmamba..."
    conda config --set solver libmamba >/dev/null 2>&1 || true

    local new_solver=""
    new_solver="$(conda config --show solver 2>/dev/null | awk '/solver:/ {print $2}' || true)"
    if [[ "${new_solver}" != "libmamba" ]]; then
        echo "Failed to configure conda to use the libmamba solver." >&2
        exit 1
    fi
}

accept_isaacsim_eula() {
    local eula_marker="${ISAACSIM_DIR}/.eula_accepted"

    if [[ -f "${eula_marker}" ]]; then
        return
    fi

    echo "Accepting Isaac Sim additional software terms for automated setup."
    touch "${eula_marker}"
}

ensure_isaacsim_checkout() {
    if [[ -f "${ISAACSIM_DIR}/build.sh" ]]; then
        return
    fi

    git -C "${REPO_ROOT}" submodule update --init --checkout third_party/IsaacSim

    if [[ ! -f "${ISAACSIM_DIR}/build.sh" ]]; then
        echo "IsaacSim submodule checkout is incomplete: ${ISAACSIM_DIR}/build.sh is missing." >&2
        echo "Check the submodule state with: git -C \"${REPO_ROOT}\" submodule status third_party/IsaacSim" >&2
        exit 1
    fi
}

ensure_rsl_rl_checkout() {
    if [[ -f "${RSL_RL_DIR}/setup.py" || -f "${RSL_RL_DIR}/pyproject.toml" ]]; then
        return
    fi

    if ! git -C "${REPO_ROOT}" submodule update --init third_party/rsl_rl; then
        echo "Failed to initialize third_party/rsl_rl." >&2
        echo "Configured URL: $(git -C "${REPO_ROOT}" config --get submodule.third_party/rsl_rl.url || echo unset)" >&2
        echo "Retry manually with: git -C \"${REPO_ROOT}\" submodule update --init third_party/rsl_rl" >&2
        echo "Then rerun this script." >&2
        exit 1
    fi
}

init_required_submodules() {
    git -C "${REPO_ROOT}" submodule update --init --recursive \
        third_party/IsaacLab \
        third_party/IsaacSim

    ensure_rsl_rl_checkout
    ensure_isaacsim_checkout
}

build_isaacsim() {
    ensure_isaacsim_checkout
    accept_isaacsim_eula

    if [[ -x "${ISAACSIM_BUILD_DIR}/isaac-sim.sh" ]]; then
        return
    fi

    if git lfs version >/dev/null 2>&1; then
        git lfs install
        if ! git -C "${ISAACSIM_DIR}" lfs pull; then
            echo "git lfs pull failed in ${ISAACSIM_DIR}." >&2
            echo "The real transport error is usually in the LFS log below." >&2
            git -C "${ISAACSIM_DIR}" lfs logs last >&2 || true
            exit 1
        fi
    fi

    if [[ ! -x /usr/bin/gcc-11 || ! -x /usr/bin/g++-11 ]]; then
        echo "IsaacSim requires GCC/G++ 11, but /usr/bin/gcc-11 or /usr/bin/g++-11 is missing." >&2
        exit 1
    fi

    (
        local compiler_shim_dir
        compiler_shim_dir="$(mktemp -d)"
        trap 'rm -rf "${compiler_shim_dir}"' EXIT

        ln -sf /usr/bin/gcc-11 "${compiler_shim_dir}/gcc"
        ln -sf /usr/bin/g++-11 "${compiler_shim_dir}/g++"
        ln -sf /usr/bin/gcc-11 "${compiler_shim_dir}/cc"
        ln -sf /usr/bin/g++-11 "${compiler_shim_dir}/c++"

        cd "${ISAACSIM_DIR}"
        PATH="${compiler_shim_dir}:${PATH}" bash ./build.sh -r -j "$(nproc)"
    )
}

write_source_build_conda_env() {
    # Isaac Sim source builds provide setup_python_env.sh but not the
    # setup_conda_env.sh expected by Isaac Lab's conda activation hooks.  Keep
    # the required Kit/extension paths while removing Kit's bundled stdlib so
    # the active conda interpreter continues to use its own standard library.
    cat > "${ISAACSIM_BUILD_DIR}/setup_conda_env.sh" <<'EOF'
#!/usr/bin/env bash

_isaacsim_build_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ISAAC_PATH="${_isaacsim_build_dir}"
export CARB_APP_PATH="${ISAAC_PATH}/kit"
export EXP_PATH="${ISAAC_PATH}/apps"

# shellcheck disable=SC1091
source "${ISAAC_PATH}/setup_python_env.sh"

_isaacsim_filter_pythonpath() {
    local old_ifs="${IFS}"
    local entry
    local filtered=()
    local entries=()

    IFS=: read -r -a entries <<< "${PYTHONPATH:-}"
    for entry in "${entries[@]}"; do
        case "${entry}" in
            "${ISAAC_PATH}/kit/python/lib/python3.12"|"${ISAAC_PATH}/kit/python/lib/python3.12/site-packages")
                ;;
            *)
                filtered+=("${entry}")
                ;;
        esac
    done
    IFS=:
    export PYTHONPATH="${filtered[*]}"
    IFS="${old_ifs}"
}

_isaacsim_filter_pythonpath
unset -f _isaacsim_filter_pythonpath
unset _isaacsim_build_dir
EOF
    chmod +x "${ISAACSIM_BUILD_DIR}/setup_conda_env.sh"
}

create_isaaclab_env() {
    echo "[INFO] Creating conda environment named '${ENV_NAME}'..."
    (
        cd "${ISAACLAB_DIR}"
        env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
            -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH \
            ./isaaclab.sh --conda "${ENV_NAME}"
    )

    conda install -n "${ENV_NAME}" -y "python=${PYTHON_VERSION}"
    # existing version verification follows

    local actual_python
    if ! actual_python="$(conda --no-plugins run -n "${ENV_NAME}" python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"; then
        echo "Unable to run Python in conda environment '${ENV_NAME}':" >&2
        echo "${actual_python}" >&2
        exit 1
    fi
    if [[ "${actual_python}" != "${PYTHON_VERSION}" ]]; then
        echo "Expected Python ${PYTHON_VERSION} in ${ENV_NAME}, found ${actual_python}." >&2
        exit 1
    fi
}

write_conda_runtime_hooks() {
    local conda_prefix
    conda_prefix="$(conda --no-plugins run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)' 2>/dev/null)"
    if [[ ! -d "${conda_prefix}" ]]; then
        echo "Unable to determine the path for conda environment '${ENV_NAME}'." >&2
        exit 1
    fi

    mkdir -p "${conda_prefix}/etc/conda/activate.d" "${conda_prefix}/etc/conda/deactivate.d"
    cat > "${conda_prefix}/etc/conda/activate.d/robros_isaacsim.sh" <<EOF
#!/usr/bin/env bash
: "\${_ROBROS_PREV_PYTHONPATH:=\${PYTHONPATH-}}"
: "\${_ROBROS_PREV_LD_LIBRARY_PATH:=\${LD_LIBRARY_PATH-}}"
export ISAACLAB_PATH="${ISAACLAB_DIR}"
alias isaaclab="${ISAACLAB_DIR}/isaaclab.sh"
source "${ISAACSIM_BUILD_DIR}/setup_conda_env.sh"
EOF
    cat > "${conda_prefix}/etc/conda/deactivate.d/robros_isaacsim.sh" <<'EOF'
#!/usr/bin/env bash
unalias isaaclab &>/dev/null || true
unset ISAACLAB_PATH CARB_APP_PATH EXP_PATH ISAAC_PATH RESOURCE_NAME
if [[ -v _ROBROS_PREV_PYTHONPATH ]]; then
    export PYTHONPATH="${_ROBROS_PREV_PYTHONPATH}"
    unset _ROBROS_PREV_PYTHONPATH
fi
if [[ -v _ROBROS_PREV_LD_LIBRARY_PATH ]]; then
    export LD_LIBRARY_PATH="${_ROBROS_PREV_LD_LIBRARY_PATH}"
    unset _ROBROS_PREV_LD_LIBRARY_PATH
fi
EOF
}

install_isaaclab_packages() {
    (
        cd "${ISAACLAB_DIR}"
        # Install every core package and let Isaac Lab apply its centralized
        # Newton/Warp/MJWarp dependency pins.  In particular, the installer
        # replaces Isaac Sim's older bundled Newton metadata with the revision
        # selected by this Isaac Lab checkout.
        # Run the CLI directly for installation so pip starts with a clean
        # Python path; the source-build runtime paths are added only by conda
        # activation hooks when the user runs Isaac Lab.
        env -u PYTHONHOME -u PYTHONSTARTUP -u PYTHONEXECUTABLE \
            PYTHONPATH="${ISAACLAB_DIR}/source/isaaclab" \
            conda run --no-capture-output -n "${ENV_NAME}" python -c \
                'from isaaclab.cli import cli; cli()' --install core
    )
}

main() {
    ensure_apt_packages
    ensure_conda

    init_required_submodules
    build_isaacsim
    write_source_build_conda_env

    if [[ -e "${ISAACLAB_DIR}/_isaac_sim" && ! -L "${ISAACLAB_DIR}/_isaac_sim" ]]; then
        echo "${ISAACLAB_DIR}/_isaac_sim already exists and is not a symlink." >&2
        exit 1
    fi
    ln -sfn "${ISAACSIM_BUILD_DIR}" "${ISAACLAB_DIR}/_isaac_sim"

    source_conda
    conda_activate_safe base
    accept_conda_tos
    ensure_conda_libmamba

    create_isaaclab_env
    write_conda_runtime_hooks

    conda run --no-capture-output -n "${ENV_NAME}" python -m pip install --upgrade pip

    install_isaaclab_packages

    conda run --no-capture-output -n "${ENV_NAME}" python -m pip install -e "${RSL_RL_DIR}"
    conda run --no-capture-output -n "${ENV_NAME}" python -m pip install -e "${ROBROS_LAB_DIR}"

    echo "Setup complete."
    echo "Activate with:"
    echo "  source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate ${ENV_NAME}"
}

main "$@"
