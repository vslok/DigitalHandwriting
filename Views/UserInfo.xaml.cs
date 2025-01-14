using DigitalHandwriting.ViewModels;
using System.Windows;

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
