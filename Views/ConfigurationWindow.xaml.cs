using DigitalHandwriting.ViewModels;
using DigitalHandwriting.Services;
using System.Windows;
using Microsoft.Extensions.DependencyInjection;

namespace DigitalHandwriting.Views
{
    public partial class ConfigurationWindow : Window
    {
        public ConfigurationWindow()
        {
            InitializeComponent();
            var settingsService = App.AppHost.Services.GetRequiredService<SettingsService>();
            var viewModel = new ConfigurationViewModel(settingsService);
            viewModel.CloseAction = new System.Action(this.Close);
            DataContext = viewModel;
        }
    }
}
