using DigitalHandwriting.ViewModels;
using DigitalHandwriting.Services;
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
using Microsoft.Extensions.DependencyInjection; // For GetRequiredService

namespace DigitalHandwriting.Views
{
    /// <summary>
    /// Interaction logic for ValidationResult.xaml
    /// </summary>
    public partial class AdministrationPanel : Window
    {
        public AdministrationPanel()
        {
            InitializeComponent();
            // Resolve DataMigrationService from the DI container
            var dataMigrationService = App.AppHost.Services.GetRequiredService<DataMigrationService>();
            // If AlgorithmValidationService were also DI-managed:
            // var algorithmValidationService = App.AppHost.Services.GetRequiredService<AlgorithmValidationService>();
            // DataContext = new AdministrationPanelViewModel(dataMigrationService, algorithmValidationService);
            DataContext = new AdministrationPanelViewModel(dataMigrationService);
        }
    }
}
