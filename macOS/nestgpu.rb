cask 'nestgpu' do
  version '1.2.2'
  sha256 :no_check

  url "http://0.0.0.0:8000/NESTGPU-macOS-1.2.0.tgz"
  name 'Neurongpu'
  homepage 'https://github.com/golosio/NESTGPU/wiki'

  depends_on macos: [
                      :sierra,
                      :high_sierra,
                    ]

  depends_on formula: 'libomp'

  depends_on formula: 'openmpi'

  installer script: {
                      executable: "#{staged_path}/NESTGPU/macOS/install.sh",
                      args: ["#{staged_path}"],
                      sudo:       true,
                    }
  uninstall script: {
                      executable: "#{staged_path}/NESTGPU/macOS/uninstall.sh",
                      sudo:       true,
                    }
  caveats "If you have not already done so, for using this software"
  caveats "you have to install NVIDIA Web Drivers for your version of macOS,"
  caveats "either from this site: https://www.tonymacx86.com/nvidia-drivers/"
  caveats "or using homebrew:"
  caveats "brew cask install nvidia-web-driver"

end
