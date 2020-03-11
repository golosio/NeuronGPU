cask 'neurongpu' do
  version '1.0.8'
  sha256 :no_check

  url "http://0.0.0.0:8000/NeuronGPU-macOS-1.0.8-alpha.2.tgz"
  name 'Neurongpu'
  homepage 'https://github.com/golosio/NeuronGPU'

  depends_on macos: [
                      :sierra,
                      :high_sierra,
                    ]
  depends_on cask: 'nvidia-cuda'

  depends_on formula: 'libomp'

  depends_on formula: 'openmpi'

  depends_on formula: 'mpi4py'

  installer script: {
                      executable: "#{staged_path}/NeuronGPU/macOS/install.sh",
                      args: ["#{staged_path}"],
                      sudo:       true,
                    }
  uninstall script: {
                      executable: "#{staged_path}/NeuronGPU/macOS/uninstall.sh",
                      sudo:       true,
                    }
  caveats "If you have not already done so, for using this software"
  caveats "you have to install NVIDIA Web Drivers for your version of macOS,"
  caveats "either from this site: https://www.tonymacx86.com/nvidia-drivers/"
  caveats "or using homebrew:"
  caveats "brew cask install nvidia-web-driver"

end
