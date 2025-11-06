package main

import (
	"encoding/json"
	"gotendance/collator"
	"html/template"
	"io"
	"log"
	"net/http"
	"strconv"
	"time"
)

var recordFilename = "output.json"

func generateOKRes() map[string]string {
	return map[string]string{
		"status": "ok",
	}
}

func loadData(store *collator.Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		err := r.ParseMultipartForm(10 << 20)
		if err != nil {
			http.Error(w, "Unable to parse form", http.StatusBadRequest)
			return
		}

		file, _, err := r.FormFile("jsonFile")
		if err != nil {
			http.Error(w, "Unable to get file from form", http.StatusBadRequest)
			return
		}

		defer file.Close()

		bytes, err := io.ReadAll(file)
		if err != nil {
			http.Error(w, "Unable to read file", http.StatusInternalServerError)
			return
		}

		err = store.LoadJSON(bytes)
		if err != nil {
			http.Error(w, "Invalid JSON format", http.StatusInternalServerError)
			return
		}

		log.Printf("Loaded data from JSON file")

		response := generateOKRes()
		w.Header().Set("Content-Type", "application/json")

		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, "Error encoding JSON response", http.StatusInternalServerError)
		}
	}
}

func startCollateHandler(store *collator.Store, streamsList *collator.StreamsList) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		err := r.ParseMultipartForm(500)
		if err != nil {
			http.Error(w, "Could not parse form", http.StatusBadRequest)
			return
		}

		updateIntervalStr := r.FormValue("updateInterval")
		floatValue, err := strconv.ParseFloat(updateIntervalStr, 64)
		if err != nil {
			http.Error(w, "Update Interval unable to be parsed", http.StatusBadRequest)
			return
		}

		updateInterval := time.Duration(floatValue * 1000)
		//log.Printf("%v", updateInterval)

		url := r.FormValue("frUrl")

		// Testing the URL
		resp, err := http.Get(url)
		if err != nil {
			http.Error(w, "URL provided does not work!", http.StatusBadRequest)
			return
		}
		resp.Body.Close()

		stopChan := streamsList.AddStreamSrc(url, updateInterval)
		if stopChan != nil {
			go collator.Stream(store, stopChan, url, updateInterval)
		}

		response := generateOKRes()
		w.Header().Set("Content-Type", "application/json")

		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, "Error encoding JSON response", http.StatusInternalServerError)
		}
	}
}

func stopCollateHandler(streamsList *collator.StreamsList) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		queryParams := r.URL.Query()

		url := queryParams.Get("frUrl")

		streamsList.RemoveStreamSrc(url)

		response := generateOKRes()
		w.Header().Set("Content-Type", "application/json")

		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, "Error encoding JSON response", http.StatusInternalServerError)
		}
	}
}

func fetchHandler(store *collator.Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET")

		jsonData, err := store.JsonOut()
		if err != nil {
			log.Printf("Error marshaling to JSON: %v", err)
			http.Error(w, "Error marshaling to JSON", http.StatusInternalServerError)
			return
		}

		store.JsonSave(recordFilename)

		w.Header().Set("Content-Type", "application/json")
		w.Write(jsonData)
	}
}

func changeAttendanceHandler(store *collator.Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST")

		queryParams := r.URL.Query()
		name := queryParams.Get("name")
		store.Mark(name)

		response := generateOKRes()
		w.Header().Set("Content-Type", "application/json")

		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, "Error encoding JSON response", http.StatusInternalServerError)
		}
	}
}

func getCountHandler(store *collator.Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET")

		response := store.Count()
		w.Header().Set("Content-Type", "application/json")

		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, "Error encoding JSON response", http.StatusInternalServerError)
		}
	}
}

func recordHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "templates/records.html")
	}
}

func homeHandler(streamsList *collator.StreamsList) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		tmpl, err := template.ParseFiles("templates/home.html")
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		streamsListCopy := streamsList.FetchList()

		err = tmpl.Execute(w, streamsListCopy)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

// Middlewares
func enforceGet(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method Not Allowed! Use Get Method", http.StatusBadRequest)
			return
		}

		log.Printf("%s %s", r.Method, r.URL.String())
		next.ServeHTTP(w, r)
	}
}

func main() {
	log.Printf("Initialising...\n")

	streamsList := collator.NewStreamsList()
	store := collator.NewStore()
	store.LoadPrevOutput(recordFilename)

	http.HandleFunc("/initData", loadData(store))

	http.HandleFunc("/startCollate", startCollateHandler(store, streamsList))
	http.HandleFunc("/stopCollate", stopCollateHandler(streamsList))

	http.HandleFunc("/changeAttendance", changeAttendanceHandler(store))
	http.HandleFunc("/fetchAttendance", fetchHandler(store))
	http.HandleFunc("/getCount", getCountHandler(store))

	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
	http.HandleFunc("/records", enforceGet(recordHandler()))
	http.HandleFunc("/", enforceGet(homeHandler(streamsList)))

	log.Printf("Starting server on :1500")
	log.Printf("Click here -> http://localhost:1500")
	http.ListenAndServe(":1500", nil)
}
