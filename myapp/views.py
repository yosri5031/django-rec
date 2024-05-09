from django.shortcuts import render
from .classifier import classify_text
from .recommendations import recommend_products,preprocess_text, generate_plot

def recommendation(request):
    if request.method == 'POST':
        # Collect user answers from the form
        answer1 = request.POST.get('occasions')
        answer2 = request.POST.get('emotions')
        answer3 = request.POST.get('audience')
        answer4 = request.POST.get('interests')
        answer5 = request.POST.get('personality')



        # Combine user answers into a single text
        user_answers = f"{answer1} {answer2} {answer3} {answer4} {answer5}"

        # Preprocess user answers
        preprocessed_answers = preprocess_text(user_answers)

        # Perform recommendation based on user answers
        recommended_products, similarity_values = recommend_products(preprocessed_answers)

        # Generate the plot
        generate_plot(similarity_values, recommended_products)

        # Render the result template with the recommended products and plot
        return render(request, 'result.html', {
            'recommended_products': recommended_products,
        })
    else:
        return render(request, 'recom.html')
    
def classify_view(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        predicted_labels = classify_text(text)
        return render(request, 'result.html', {'labels': predicted_labels})
    return render(request, 'index.html')