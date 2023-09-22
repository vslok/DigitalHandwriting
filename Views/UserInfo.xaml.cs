using DigitalHandwriting.Models;
using DigitalHandwriting.ViewModels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace DigitalHandwriting.Views
{
    /// <summary>
    /// Interaction logic for UserInfo.xaml
    /// </summary>
    public partial class UserInfo : UserControl
    {
        UserInfoViewModel _viewModel;
        public UserInfo(User user, List<int> keyPressedTimes, List<int> beetwenKeysTimes)
        {
            InitializeComponent();
            _viewModel = new UserInfoViewModel(user, keyPressedTimes, beetwenKeysTimes);
            DataContext = _viewModel;
        }
    }
}
