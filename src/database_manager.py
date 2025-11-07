def insert_result(self, queue_id: str, sentiment_label: str, 
                 confidence_score: float, json_result: dict, processed_by: str):
    """Insert analysis result"""
    with self.get_session() as session:
        try:
            result = OutputResult(
                queue_id=queue_id,
                sentiment_label=sentiment_label,
                confidence_score=confidence_score,
                json_result=json_result,
                processed_by=processed_by
            )
            session.add(result)
            session.commit()
            logger.info(f"✅ Inserted result for queue {queue_id}")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"❌ Failed to insert result: {e}")
            return False
