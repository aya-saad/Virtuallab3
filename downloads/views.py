# Create your views here.
from django.http import HttpResponse
from django.template import loader
from models.models import Experiment, Download, ExpDownload
from django.shortcuts import get_object_or_404, render
from django.urls import reverse


def index(request):
    template = loader.get_template('downloads.html')
    exp_list = Experiment.objects.order_by('exp_num')[:]
    download_list = Download.objects.order_by('download_num')[:]
    context = {
        'exp_list': exp_list,
        'download_list': download_list,
    }
    return HttpResponse(template.render(context, request))
