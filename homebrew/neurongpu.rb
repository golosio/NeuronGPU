class neurongpu < Formula
  desc "A GPU library for simulation of large scale networks of biological neurons."
  homepage "https://github.com/golosio/NeuronGPU"
  url "https://github.com/golosio/NeuronGPU/archive/master.zip"
  version "1.0.8
  #sha256 ""
  depends_on "autoconf" => :build
  depends_on "automake" => :build
  #depends_on ""

  def install
    system "autoreconf", "-i"
    system "./configure", "--prefix=#{prefix}"
    system "make", "install"
  end

  def caveats
    s = <<~EOS
    Prova:
    EOS
    s
  end

end
