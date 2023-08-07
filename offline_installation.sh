#!/bin/bash

# Ensure that the script is run as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root."
    exit 1
fi

# Directory where the offline packages are stored
offline_dir="offline_install"

# Function to install packages from the offline directory
install_packages() {
    for pkg in "$@"; do
        echo "Installing ${pkg}..."
        dpkg -i "${offline_dir}/${pkg}"
        if [ $? -ne 0 ]; then
            echo "Failed to install ${pkg}. Please check dependencies."
            exit 1
        fi
    done
}

# List of package filenames
packages=("libbsd0_0.11.5-1_amd64.deb"
          "liblzma5_5.2.5-2ubuntu1_amd64.deb"
          "libselinux1_3.3-1build2_amd64.deb"
          "zlib1g_1%3a1.2.11.dfsg-2ubuntu9.2_amd64.deb"
          "libacl1_2.3.1-1_amd64.deb"
	  "gcc-12-base_12.3.0-1ubuntu1~22.04_amd64.deb"
          "libgcc-s1_12.3.0-1ubuntu1~22.04_amd64.deb"
          "libmd0_1.0.4-1build1_amd64.deb"
          "libssl3_3.0.2-0ubuntu1.10_amd64.deb"
          "libxdmcp6_1%3a1.1.3-0ubuntu5_amd64.deb"
          "libc6_2.35-0ubuntu3.1_amd64.deb"
          "libjudydebian1_1.0.5-5_amd64.deb"
          "libsctp1_1.0.19+dfsg-1build1_amd64.deb"
          "libx11-data_2%3a1.7.5-1ubuntu0.2_all.deb"
          "libslang2_2.3.2-5build4_amd64.deb"
          "libipsec-mb1_1.2-1_amd64.deb"
          "libxcb1_1.14-3ubuntu3_amd64.deb"
          "debconf_1.5.79ubuntu1_all.deb"
          "libzstd1_1.4.8+dfsg-3build1_amd64.deb"
          "libdebian-installer4_0.122ubuntu3_amd64.deb"
          "dpkg_1.21.1ubuntu2.2_amd64.deb"
          "libbz2-1.0_1.0.8-5build1_amd64.deb"
          "libjpeg8_8c-2ubuntu10_amd64.deb"
          "libpcre2-8-0_10.39-3ubuntu0.1_amd64.deb"
          "libtextwrap1_0.1-15build1_amd64.deb"
          "libseccomp2_2.5.3-2ubuntu2_amd64.deb"
          "libaudit1_1%3a3.0.7-1build1_amd64.deb"
          "libcap-ng0_0.7.9-2.2build3_amd64.deb"
          "libcrypt1_1%3a4.4.27-1_amd64.deb"
          "libcom-err2_1.46.5-2ubuntu1.1_amd64.deb"
          "libgmp10_2%3a6.2.1+dfsg-3ubuntu1_amd64.deb"
          "libselinux1_3.3-1build2_amd64.deb"
          "libbz2-1.0_1.0.8-5build1_amd64.deb"
          "liblzma5_5.2.5-2ubuntu1_amd64.deb"
          "zlib1g_1%3a1.2.11.dfsg-2ubuntu9.2_amd64.deb"
          "glmark2-data_2021.02-0ubuntu1_all.deb"
          "glmark2_2021.02-0ubuntu1_amd64.deb"
          "stress-ng_0.13.12-2ubuntu1_amd64.deb"
          "cdebconf_0.261ubuntu1_amd64.deb")

# Define dependencies for each package
declare -A dependencies
dependencies["cdebconf_0.261ubuntu1_amd64.deb"]=""
dependencies["debconf_1.5.79ubuntu1_all.deb"]=""
dependencies["libapparmor1_3.0.4-2ubuntu2.2_amd64.deb"]=""
dependencies["libipsec-mb1_1.2-1_amd64.deb"]=""
dependencies["libnewt0.52_0.52.21-5ubuntu2_amd64.deb"]=""
dependencies["libstdc++6_12.3.0-1ubuntu1~22.04_amd64.deb"]=""
dependencies["libxxhash0_0.8.1-1_amd64.deb"]=""
dependencies["libjpeg8_8c-2ubuntu10_amd64.deb"]=""
dependencies["libpcre2-8-0_10.39-3ubuntu0.1_amd64.deb"]=""
dependencies["libtextwrap1_0.1-15build1_amd64.deb"]=""
dependencies["libzstd1_1.4.8+dfsg-3build1_amd64.deb"]=""
dependencies["dpkg_1.21.1ubuntu2.2_amd64.deb"]=""
dependencies["libbz2-1.0_1.0.8-5build1_amd64.deb"]=""
dependencies["libjpeg-turbo8_2.1.2-0ubuntu1_amd64.deb"]=""
dependencies["libpng16-16_1.6.37-3build5_amd64.deb"]=""
dependencies["libx11-6_2%3a1.7.5-1ubuntu0.2_amd64.deb"]=""
dependencies["perl-base_5.34.0-3ubuntu1.2_amd64.deb"]=""
dependencies["gcc-12-base_12.3.0-1ubuntu1~22.04_amd64.deb"]=""
dependencies["libc6_2.35-0ubuntu3.1_amd64.deb"]=""
dependencies["libkmod2_29-1ubuntu1_amd64.deb"]=""
dependencies["libselinux1_3.3-1build2_amd64.deb"]=""
dependencies["libbz2-1.0_1.0.8-5build1_amd64.deb"]=""
dependencies["liblzma5_5.2.5-2ubuntu1_amd64.deb"]=""
dependencies["zlib1g_1%3a1.2.11.dfsg-2ubuntu9.2_amd64.deb"]=""
dependencies["libacl1_2.3.1-1_amd64.deb"]=""
dependencies["libgcc-s1_12.3.0-1ubuntu1~22.04_amd64.deb"]=""
dependencies["libmd0_1.0.4-1build1_amd64.deb"]=""
dependencies["libssl3_3.0.2-0ubuntu1.10_amd64.deb"]=""
dependencies["libxdmcp6_1%3a1.1.3-0ubuntu5_amd64.deb"]=""
dependencies["libc6_2.35-0ubuntu3.1_amd64.deb"]=""
dependencies["libjudydebian1_1.0.5-5_amd64.deb"]=""
dependencies["libsctp1_1.0.19+dfsg-1build1_amd64.deb"]=""
dependencies["libx11-data_2%3a1.7.5-1ubuntu0.2_all.deb"]=""
dependencies["libslang2_2.3.2-5build4_amd64.deb"]=""
dependencies["libxcb1_1.14-3ubuntu3_amd64.deb"]=""
dependencies["zlib1g_1%3a1.2.11.dfsg-2ubuntu9.2_amd64.deb"]=""
dependencies["glmark2-data_2021.02-0ubuntu1_all.deb"]=""
dependencies["glmark2_2021.02-0ubuntu1_amd64.deb"]=""
dependencies["stress-ng_0.13.12-2ubuntu1_amd64.deb"]=""

# Install packages and their dependencies
for pkg in "${packages[@]}"; do
    # Install dependencies first
    for dependency in ${dependencies["$pkg"]}; do
        install_packages "$dependency"
    done
    # Then install the package
    install_packages "$pkg"
done

echo "All packages installed successfully."
