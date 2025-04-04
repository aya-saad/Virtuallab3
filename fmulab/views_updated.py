"""
Updated views for FMU simulation with Graph LLM integration
"""
import os
import json
import logging
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views import generic
from django.shortcuts import get_object_or_404, render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages

from models.models import Experiment
from .models_updated import FMUForm, ChatSession, ChatMessage
from .graph_db import Neo4jConnection
from .llm_integration import get_llm_provider
from .qa_integration import QAIntegration
from .config import CHAT_MODE_CONFIG_MAP, CHAT_DEFAULT_MODE

# Initialize the QA pipeline
qa_pipeline = None


def get_qa_pipeline():
    """Get or initialize the QA pipeline"""
    global qa_pipeline
    if qa_pipeline is None:
        neo4j_conn = Neo4jConnection(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            username=os.environ.get("NEO4J_USERNAME", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password")
        )
        neo4j_conn.connect()
        llm_provider = get_llm_provider(
            provider=os.environ.get("LLM_PROVIDER", "openai"),
            model=os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        )
        qa_pipeline = QAIntegration(neo4j_conn, llm_provider)
    return qa_pipeline


def index(request):
    """Main FMU Lab index page with chat interface"""
    template = loader.get_template('fmulab/index.html')

    # Get experiments for the sidebar
    exp_list = Experiment.objects.order_by('exp_num')[:]

    # Get available documents for search
    qa = get_qa_pipeline()
    available_documents = qa.get_documents()

    # Create document choices for the form
    document_choices = [(doc, doc) for doc in available_documents]

    # Initialize the form
    form = FMUForm()
    form.fields['document_selector'].choices = document_choices

    # Get or create session for this user
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = f"session_{request.session.session_key}"
        request.session['chat_session_id'] = session_id

    # Get chat history
    try:
        chat_session, created = ChatSession.objects.get_or_create(session_id=session_id)
        chat_history = ChatMessage.objects.filter(session=chat_session).order_by('created_at')
    except Exception as e:
        logging.error(f"Error retrieving chat session: {str(e)}")
        chat_history = []

    context = {
        'exp_list': exp_list,
        'form': form,
        'chat_history': chat_history,
        'available_documents': available_documents,
        'chat_modes': CHAT_MODE_CONFIG_MAP
    }

    return HttpResponse(template.render(context, request))


def home(request):
    """Home view - simplified here for compatibility"""
    template = loader.get_template('fmulab/home.html')
    return HttpResponse(template.render({}, request))


def dashboard(request):
    """Process form submission and get LLM response"""
    if request.method == "POST":
        # Get the form data
        form = FMUForm(request.POST)

        if form.is_valid():
            # Extract the question and settings
            question = form.cleaned_data['exp_desc']
            selected_documents = form.cleaned_data.get('document_selector', [])
            chat_mode = form.cleaned_data.get('chat_mode', CHAT_DEFAULT_MODE)

            # Get or create session for this user
            session_id = request.session.get('chat_session_id')
            if not session_id:
                session_id = f"session_{request.session.session_key}"
                request.session['chat_session_id'] = session_id

            try:
                # Get the session from DB
                chat_session, created = ChatSession.objects.get_or_create(session_id=session_id)

                # Save the user message
                user_message = ChatMessage.objects.create(
                    session=chat_session,
                    role='user',
                    content=question
                )

                # Get response from QA pipeline
                qa = get_qa_pipeline()
                response = qa.get_chat_response(
                    query=question,
                    session_id=session_id,
                    document_names=selected_documents if selected_documents else None
                )

                # Save the assistant message
                assistant_message = ChatMessage.objects.create(
                    session=chat_session,
                    role='assistant',
                    content=response['message'],
                    sources=json.dumps(response.get('sources', []))
                )

                # Redirect back to the chat interface
                return redirect('fmulab:index')

            except Exception as e:
                logging.error(f"Error processing question: {str(e)}")
                messages.error(request, f"Error processing question: {str(e)}")
                return redirect('fmulab:index')
        else:
            # Form is invalid, show errors
            return render(request, 'fmulab/index.html', {'form': form})
    else:
        # GET request - redirect to index
        return redirect('fmulab:index')


@csrf_exempt
def chat_api(request):
    """API endpoint for AJAX chat functionality"""
    if request.method == "POST":
        try:
            # Parse the request data
            data = json.loads(request.body)
            question = data.get('message', '')
            selected_documents = data.get('documents', [])
            chat_mode = data.get('mode', CHAT_DEFAULT_MODE)

            # Get or create session for this user
            session_id = request.session.get('chat_session_id')
            if not session_id:
                session_id = f"session_{request.session.session_key}"
                request.session['chat_session_id'] = session_id

            # Get response from QA pipeline
            qa = get_qa_pipeline()
            response = qa.get_chat_response(
                query=question,
                session_id=session_id,
                document_names=selected_documents if selected_documents else None
            )

            # Get or create the chat session
            chat_session, created = ChatSession.objects.get_or_create(session_id=session_id)

            # Save the user message
            user_message = ChatMessage.objects.create(
                session=chat_session,
                role='user',
                content=question
            )

            # Save the assistant message
            assistant_message = ChatMessage.objects.create(
                session=chat_session,
                role='assistant',
                content=response['message'],
                sources=json.dumps(response.get('sources', []))
            )

            # Return the response
            return JsonResponse({
                'message': response['message'],
                'sources': response.get('sources', []),
                'session_id': session_id
            })

        except Exception as e:
            logging.error(f"Error in chat API: {str(e)}")
            return JsonResponse({
                'error': str(e)
            }, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def clear_chat(request):
    """Clear the chat history"""
    session_id = request.session.get('chat_session_id')
    if session_id:
        try:
            # Find the chat session
            chat_session = ChatSession.objects.get(session_id=session_id)

            # Delete all messages
            chat_session.messages.all().delete()

            messages.success(request, "Chat history cleared.")
        except Exception as e:
            logging.error(f"Error clearing chat: {str(e)}")
            messages.error(request, f"Error clearing chat: {str(e)}")

    return redirect('fmulab:index')