import markdown

def cleanse_response(old_response):
    answerForeword = "Answer:"
    answer_start = old_response.find(answerForeword) + len(answerForeword)
    new_response = old_response[answer_start:].replace("<|assistant|>", "").strip()
    return new_response


def markdown_to_html(md_text):
    return markdown.markdown(
        md_text,
        extensions=['extra', 'nl2br'] 
    )