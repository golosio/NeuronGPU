sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo ubuntu-drivers autoinstall
echo
echo "Done. Before continuing with the installation you must reboot the system."
echo "After reboot, open a terminal, cd to this same folder and type:"
echo "./install.sh"
read -p "Do you want to reboot now [yN]? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    sudo reboot
fi

