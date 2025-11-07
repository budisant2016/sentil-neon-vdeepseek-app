# Dalam class SentilBackend, update process_queue method:
def process_queue(self):
    """Main method to process queued items"""
    logger.info("🔄 Starting queue processing...")
    
    queued_items = self.db.get_queued_items(self.processing_batch_size)
    
    if not queued_items:
        logger.info("ℹ️ No items in queue")
        return 0
    
    processed_count = 0
    
    for item in queued_items:
        try:
            logger.info(f"📥 Processing item {item['queue_id']} for user {item['user_id']}")
            
            # Acquire session slot
            slot_id = self.db.acquire_session_slot(item['tier'], item['user_id'])
            
            if not slot_id:
                logger.warning(f"⚠️ No available slot for tier {item['tier']}")
                continue
            
            # Update status to processing
            self.db.update_queue_status(item['queue_id'], 'processing', slot_id)
            self.db.log_system_activity('backend', f'Started processing {item["queue_id"]}', 'info', item['queue_id'])
            
            # Perform sentiment analysis dengan auto language detection
            result = self.analyzer.analyze_sentiment(
                item['input_text'], 
                item.get('method', 'NaiveBayes'),
                language='auto'  # Auto-detect language
            )
            
            # Save result
            self.db.insert_result(
                queue_id=item['queue_id'],
                sentiment_label=result['sentiment_label'],
                confidence_score=result['confidence_score'],
                json_result=result,
                processed_by=f"Streamlit_{item.get('method', 'NaiveBayes')}_{result['language_detected']}"
            )
            
            # Update queue status to done
            self.db.update_queue_status(item['queue_id'], 'done')
            self.db.release_session_slot(slot_id)
            
            self.db.log_system_activity('backend', f'Completed processing {item["queue_id"]}', 'info', item['queue_id'])
            processed_count += 1
            
            logger.info(f"✅ Successfully processed {item['queue_id']} - Language: {result['language_detected']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {item['queue_id']}: {e}")
            self.db.update_queue_status(item['queue_id'], 'error')
            self.db.log_system_activity('backend', f'Error processing {item["queue_id"]}: {str(e)}', 'error', item['queue_id'])
    
    return processed_count

# Di bagian Manual Test, tambahkan language selection:
if st.session_state.get('show_test', False):
    st.subheader("🧪 Manual Test Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        test_text = st.text_area("Test text:", "I love this product! It's amazing!")
    with col2:
        test_method = st.selectbox("Method", ['NaiveBayes', 'KNN', 'RandomForest', 'SVM'])
        test_language = st.selectbox("Language", ['auto', 'english', 'indonesian'], 
                                   help="Auto: detect automatically, English/Indonesian: force specific language")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Test", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    result = backend.analyzer.analyze_sentiment(test_text, test_method, test_language)
                    
                    st.subheader("📊 Results")
                    
                    # Display main results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        sentiment_color = "🟢" if result['sentiment_label'] == 'positive' else "🔴" if result['sentiment_label'] == 'negative' else "🟡"
                        st.metric("Sentiment", f"{sentiment_color} {result['sentiment_label']}")
                    with col2:
                        st.metric("Confidence", f"{result['confidence_score']:.2%}")
                    with col3:
                        st.metric("Method", result['method_used'])
                    with col4:
                        lang_flag = "🇺🇸" if result['language_detected'] == 'english' else "🇮🇩"
                        st.metric("Language", f"{lang_flag} {result['language_detected']}")
                    
                    # Additional info
                    with st.expander("Detailed Results"):
                        st.json(result)
                        
                    # Language detection info
                    st.info(f"**Language Detection:** {result['language_detected'].upper()} - {result['processed_text']}")
                        
                except Exception as e:
                    st.error(f"Test failed: {e}")
    
    with col2:
        if st.button("Close Test"):
            st.session_state.show_test = False
            st.rerun()

# Tambahkan contoh text untuk testing
st.sidebar.subheader("🌐 Test Examples")
if st.sidebar.button("English Example"):
    st.session_state.test_text = "I absolutely love this product! It's fantastic and works perfectly."
    st.session_state.show_test = True
    st.rerun()

if st.sidebar.button("Indonesian Example"):
    st.session_state.test_text = "Saya sangat suka produk ini! Kualitasnya bagus dan harganya terjangkau."
    st.session_state.show_test = True
    st.rerun()

if st.sidebar.button("Mixed Example"):
    st.session_state.test_text = "Produknya okay, but could be better. Cukup lumayan untuk harganya."
    st.session_state.show_test = True
    st.rerun()
