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
    /// Interaction logic for ValidationResult.xaml
    /// </summary>
    public partial class ValidationResult : Window
    {
        AlgorithmValidationViewModel _viewModel;
        public ValidationResult(string testDataPath)
        {
            InitializeComponent();
            _viewModel = new AlgorithmValidationViewModel(testDataPath);
            DataContext = _viewModel;
        }
    }
}
