#
# Author: Stephan Janssen
# Date: 2022-12-19
#
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import datetime
import pickle 
import pandas as pd

model = SentenceTransformer("flax-sentence-embeddings/all_datasets_v4_MiniLM-L6") 


class Book:
    """Book class"""
    def __init__(self, title, year, author="", description="", thumb="", isbn13=""):
        self.title = title
        self.author = author
        self.description = description
        self.year = year        
        self.thumb = thumb
        self.isbn13 = isbn13

    def __str__(self):
        return f'{self.title} by {self.author} ({self.year})\n{self.isbn13}\n{self.description}'    


def getBookInfo(domain, book_url, year):
    """ Get book info """
    page = requests.get(domain + book_url)    

    # Parse the HTML content
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # title, author, description, year, category, image
    title = soup.find(class_='bookmain').find('h1').text

    table = soup.find(class_="table table-striped")
    book_elements = table.findAll('tr')

    if len(book_elements) == 10:
        authors = book_elements[2].find('b').text        
    else:
        authors = book_elements[3].find('b').text
        
    isbn13 = soup.find(id='isbn13').text

    description = soup.find(id="desc").text

    thumb = domain + "/img/books/" + isbn13 + ".png"

    return Book(title, year=year, author=authors, description=description, isbn13=isbn13, thumb=thumb)



# Get books per category using paging
def getBooksPerYear(domain, year, books):
    """Get books per year"""

    # Get books per year https://itbook.store/books/2022
    page = requests.get(domain + "/books/" + str(year))

    # Parse the HTML content
    soup = BeautifulSoup(page.content, 'html.parser')

    # Loop over books
    found_books = soup.findAll(class_='row mb10')

    # Print the text of each book
    for book in found_books:
        
        a_book = book.find('a')
        books_ref = a_book['href']
        
        book = getBookInfo(domain, books_ref, year)
        
        books.append(book)

        

def main():

    # Get year of today
    today = datetime.date.today()
    until_year = today.year

    books = []

    for (year) in range(1994, 1996):    # should be until_year 
        getBooksPerYear("https://itbook.store", year, books)        

    # Create a dataframe from the list of books
    df = pd.DataFrame(books)

    # Specify the columns of the dataframe and their corresponding values
    df = pd.DataFrame({
        "title": [book.title for book in books],
        "author": [book.author for book in books],
        "year": [book.year for book in books],
        "description": [book.description for book in books],
        "thumb": [book.thumb for book in books],
        "isbn13": [book.isbn13 for book in books]        
    })    

    df = df.assign(combined=lambda x: x.title.str.strip() + " " + x.description.str.strip()) 
    
    df["embedding"] = df.combined.apply(lambda x: model.encode(x)) 

    # Store pickle to disk
    with open('./books.pkl', 'wb') as f:
        pickle.dump(df, f)


# Call the main method
if __name__ == "__main__":
    main()