from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader
from .models import Experiment, XML
from django.shortcuts import get_object_or_404, render
from django.urls import reverse


def index(request):
    template = loader.get_template('models.html')
    exp_list = Experiment.objects.order_by('exp_num')[:]
    context = {
        'exp_list': exp_list,
    }
    return HttpResponse(template.render(context, request))


'''
def detail(request, exp_num):
    experiment = get_object_or_404(Experiment, pk=exp_num)

    template = loader.get_template('models/detail.html')
    exp_list = Experiment.objects.order_by('exp_num')[:]
    context = {
        'exp_list': exp_list,
        'experiment': experiment,
    }
    return HttpResponse(template.render(context, request))
'''

def detail(request, exp_num):
    template = loader.get_template('models/detail.html')
    exp_list = Experiment.objects.order_by('exp_num')[:]
    # Retrieve the experiment object using the exp_num
    experiment = Experiment.objects.get(id=exp_num)
    # Retrieve the related downloads for the experiment
    downloads = experiment.expdownload_set.all()
    # Retrieve the related xml for the experiment
    xmls = experiment.expxml_set.all()
    videos = experiment.expvideo_set.all()
    pdfs = experiment.exppdf_set.all()
    # return render(request, 'models/detail.html', {'experiment': experiment, 'downloads': downloads})
    context = {
        'exp_list': exp_list,
        'experiment': experiment,
        'downloads': downloads,
        'xmls': xmls,
        'videos': videos,
        'pdfs': pdfs,
    }
    return HttpResponse(template.render(context, request))
