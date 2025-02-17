using DigitalHandwriting.Factories.AuthenticationMethods.Models;
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
        public UserInfo(AuthenticationResult authenticationResult)
        {
            InitializeComponent();
            _viewModel = new UserInfoViewModel(authenticationResult);
            DataContext = _viewModel;
        }
    }
}
