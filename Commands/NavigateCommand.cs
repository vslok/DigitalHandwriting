using DigitalHandwriting.Services;
using DigitalHandwriting.ViewModels;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;

namespace DigitalHandwriting.Commands
{
    internal class NavigateCommand<TViewModel> : BaseCommand
        where TViewModel : BaseViewModel
    {
        private readonly NavigationService<TViewModel> _navigationService;
        private readonly Action? _actionBeforeNavigation;

        public NavigateCommand(NavigationService<TViewModel> navigationService, Action? actionBeforeNavigation = null)
        {
            _navigationService = navigationService;
            _actionBeforeNavigation = actionBeforeNavigation;
        }

        protected override void MainAction(object parameter)
        {
            _actionBeforeNavigation?.Invoke();
            _navigationService.Navigate();
        }
    }
}
