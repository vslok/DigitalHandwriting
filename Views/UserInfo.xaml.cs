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
    public partial class UserInfo : Window
    {
        UserInfoViewModel _viewModel;
        public UserInfo(bool isAuthenticated, double keyPressedMetric, double betweenKeysMetric, double betweenKeysPressMetric)
        {
            InitializeComponent();
            _viewModel = new UserInfoViewModel(isAuthenticated, keyPressedMetric, betweenKeysMetric, betweenKeysPressMetric);
            DataContext = _viewModel;
        }
    }
}
